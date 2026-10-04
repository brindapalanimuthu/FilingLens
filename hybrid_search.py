import pickle, re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

STOP = set("what was were is the a an of in as for did how much apple apples "
           "microsoft msft aapl fiscal year s did do does to and on at by".split())

COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone", "ipad"],
    "MSFT": ["microsoft", "msft", "azure", "windows", "xbox", "linkedin"],
}

# abbreviation -> wording used in the filings
TERM_ALIASES = {
    r"\br&d\b": "research and development",
    r"\beps\b": "earnings per share",
    r"\bsg&a\b": "selling general and administrative",
    r"\bcapex\b": "additions to property and equipment",
    r"\bnet profit\b": "net income",
}

# words that describe a computation, not a row label
GROWTH_WORDS = {"grow", "grew", "growth", "change", "changed", "increase",
                "increased", "decrease", "decreased", "rise", "rose", "fall", "fell"}
LABEL_NOISE = GROWTH_WORDS | {"percentage", "percent", "rate"}

PREFIX_PENALTY = 0.5   # 'Section > Label' rows rank a bit below an exact plain label

YEAR_TOKEN = re.compile(r"(19|20)\d{2}")


def detect_companies(query):
    """Return the set of tickers the question mentions, or None if it names none."""
    q = query.lower()
    found = {
        t for t, aliases in COMPANY_ALIASES.items()
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in aliases)
    }
    return found or None


def expand_terms(query):
    for pat, rep in TERM_ALIASES.items():
        query = re.sub(pat, rep, query, flags=re.I)
    return query


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def split_parts(query):
    """'R&D expense and net income in 2024' -> two sub-questions.
    Only splits on ' and ' when every part has >= 2 real words (years excluded),
    so 'cash and cash equivalents' is not split."""
    raw_parts = re.split(r"\s+and\s+", query.strip().rstrip("?"), flags=re.I)
    if len(raw_parts) < 2:
        return [query]
    parts = [expand_terms(p) for p in raw_parts]
    for p in parts:
        words = [w for w in tok(p) if not YEAR_TOKEN.fullmatch(w)]
        if len(words) < 2:
            return [query]
    return parts


def label_info(embed_text):
    """(label tokens, has_section_prefix). Row text is:
    filing | company | statement title | label | values..."""
    parts = embed_text.split(" | ")
    label = parts[3] if len(parts) > 3 else ""
    has_prefix = ">" in label
    label = label.split(">")[-1]          # 'Section > Net income' -> 'Net income'
    return set(tok(label)), has_prefix


_model = None
_state = {}


def _load(model=None):
    global _model
    if _state:
        return
    _model = model or SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.load("full_index/embeddings.npy")
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    entries = pickle.load(open("full_index/entries.pkl", "rb"))
    tidx = [i for i, e in enumerate(entries) if e["type"] == "table"]
    infos = [label_info(entries[i]["embed_text"]) for i in tidx]
    _state.update(
        entries=entries, tidx=tidx, t_emb=emb[tidx],
        t_company=np.array([entries[i]["company"] for i in tidx]),
        t_label=[inf[0] for inf in infos],
        t_weight=np.array([PREFIX_PENALTY if inf[1] else 1.0 for inf in infos]),
        bm25=BM25Okapi([tok(entries[i]["embed_text"]) for i in tidx]),
    )


def _rrf(*orders, k=60):
    score = {}
    for order in orders:
        for r, j in enumerate(order, 1):
            score[j] = score.get(j, 0) + 1 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def _rank(query, allowed):
    """Ranked list of table-row indices for ONE query (embedding + BM25 + label match)."""
    qv = _model.encode([query])[0]
    qv = qv / np.linalg.norm(qv)

    o_emb = [j for j in np.argsort(-(_state["t_emb"] @ qv)) if allowed[j]][:100]

    bm = _state["bm25"].get_scores(tok(query))
    o_bm = [j for j in np.argsort(-bm) if allowed[j]][:100]

    qt = {w for w in tok(query) if not YEAR_TOKEN.fullmatch(w)}
    if qt & GROWTH_WORDS:                 # growth question: drop computation words
        qt = qt - LABEL_NOISE
    lab = np.array([
        len(qt & l) / len(qt | l) if qt and l else 0.0
        for l in _state["t_label"]
    ]) * _state["t_weight"]
    order = np.lexsort((-bm, -lab))
    o_lab = [j for j in order if allowed[j] and lab[j] > 0][:100]

    return _rrf(o_emb, o_bm, o_lab, o_lab)


def hybrid_table_search(query, k=8, model=None, companies=None):
    """Top-k table-row entries. Multi-metric questions are also searched
    part by part and interleaved, so each metric gets rows in the result.
    companies: set of tickers to restrict to; None = auto-detect from query."""
    _load(model)
    if companies is None:
        companies = detect_companies(query)
    allowed = (np.isin(_state["t_company"], list(companies))
               if companies else np.ones(len(_state["tidx"]), dtype=bool))

    full = expand_terms(query)
    rankings = [_rank(full, allowed)]
    parts = split_parts(query)
    if len(parts) > 1:
        rankings += [_rank(p, allowed) for p in parts]

    # round-robin merge: full-question ranking first, then each part
    top, seen, depth = [], set(), 0
    while len(top) < k and depth < 100:
        for r in rankings:
            if depth < len(r) and r[depth] not in seen:
                seen.add(r[depth])
                top.append(r[depth])
                if len(top) == k:
                    break
        depth += 1

    return [_state["entries"][_state["tidx"][j]] for j in top]