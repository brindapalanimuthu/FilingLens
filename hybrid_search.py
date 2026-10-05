import math
import pickle, re
from collections import Counter
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

# --- segment / percentage re-ranking (applied after RRF) ---
SEGMENT_TERMS = [
    "iphone", "mac", "ipad", "wearables", "services",                      # Apple
    "productivity and business processes", "intelligent cloud",            # MSFT
    "more personal computing",
]
SEGMENT_BOOST = 1.5        # row label names a segment the question names
SEGMENT_PENALTY = 0.6      # row label names none of them
PCT_PENALTY = 0.5          # percentage rows when the question isn't about a share/percent
PCT_WORDS = ("percent", "%", "share", "proportion", "margin", "rate", "ratio")

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
    so 'cash and cash equivalents' is not split.
    Shared-tail case: 'iPhone and Services net sales in 2025' ->
    'iPhone net sales in 2025' + 'Services net sales in 2025'."""
    raw_parts = re.split(r"\s+and\s+", query.strip().rstrip("?"), flags=re.I)
    if len(raw_parts) < 2:
        return [query]
    parts = [expand_terms(p) for p in raw_parts]

    def words(p):
        return [w for w in tok(p) if not YEAR_TOKEN.fullmatch(w)]

    # first part is a lone label and the last part carries the shared tail
    if len(parts) == 2 and len(words(parts[0])) == 1 and len(words(parts[1])) >= 3:
        head = words(parts[0])[0]
        last_words = words(parts[1])
        if head not in last_words:      # 'cash and cash equivalents' stays whole
            tail = re.sub(rf"^.*?\b{re.escape(last_words[0])}\b", "",
                          parts[1], count=1, flags=re.I).strip()
            if tail:
                parts[0] = f"{parts[0]} {tail}"

    for p in parts:
        if len(words(p)) < 2:
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


def label_full(embed_text):
    """(full label incl. any 'Section >' prefix, lowercased; is_percentage_row)."""
    parts = embed_text.split(" | ")
    label = (parts[3] if len(parts) > 3 else "").lower()
    values = parts[4:]
    is_pct = ("percentage" in label or label.startswith("%")
              or any(v.strip() == "%" for v in values))
    return label, is_pct


def query_segments(query):
    q = query.lower()
    return [s for s in SEGMENT_TERMS if re.search(rf"\b{re.escape(s)}\b", q)]


def wants_percent(query):
    q = query.lower()
    return any(w in q for w in PCT_WORDS)


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
    fulls = [label_full(entries[i]["embed_text"]) for i in tidx]

    # rarity weight for each label word: rare words (iphone) count for more
    # than words that appear in many labels (net, sales, total)
    df = Counter(w for inf in infos for w in inf[0])
    n_labels = len(infos)
    idf = {w: math.log((n_labels + 1) / (c + 1)) + 1 for w, c in df.items()}

    _state.update(
        entries=entries, tidx=tidx, t_emb=emb[tidx],
        t_company=np.array([entries[i]["company"] for i in tidx]),
        t_label=[inf[0] for inf in infos],
        t_weight=np.array([PREFIX_PENALTY if inf[1] else 1.0 for inf in infos]),
        t_full=[f[0] for f in fulls],
        t_pct=[f[1] for f in fulls],
        bm25=BM25Okapi([tok(entries[i]["embed_text"]) for i in tidx]),
        idf=idf,
        idf_default=math.log(n_labels + 1) + 1,
    )


def _rrf_scores(*orders, k=60):
    score = {}
    for order in orders:
        for r, j in enumerate(order, 1):
            score[j] = score.get(j, 0) + 1 / (k + r)
    return score


def _rrf(*orders, k=60):
    score = _rrf_scores(*orders, k=k)
    return sorted(score, key=score.get, reverse=True)


def _row_multiplier(j, segments, pct_ok):
    m = 1.0
    if segments:
        label = _state["t_full"][j]
        m *= SEGMENT_BOOST if any(s in label for s in segments) else SEGMENT_PENALTY
    if not pct_ok and _state["t_pct"][j]:
        m *= PCT_PENALTY
    return m


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

    idf, default = _state["idf"], _state["idf_default"]

    def wt(t):
        return idf.get(t, default)

    qt_weight = sum(wt(t) for t in qt)
    lab = np.array([
        sum(wt(t) for t in qt & l) / (qt_weight + sum(wt(t) for t in l - qt))
        if qt and l else 0.0
        for l in _state["t_label"]
    ]) * _state["t_weight"]
    order = np.lexsort((-bm, -lab))
    o_lab = [j for j in order if allowed[j] and lab[j] > 0][:100]

    score = _rrf_scores(o_emb, o_bm, o_lab, o_lab)

    # re-rank: favour rows naming the segment asked about, demote % rows
    segments = query_segments(query)
    pct_ok = wants_percent(query)
    adj = {j: s * _row_multiplier(j, segments, pct_ok) for j, s in score.items()}
    return sorted(adj, key=adj.get, reverse=True)


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