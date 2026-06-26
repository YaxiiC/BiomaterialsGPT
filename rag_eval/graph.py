from collections import defaultdict

# domain terms whose co-occurrence implies a real relationship between chunks.
# co-occurrence graph is cheap and avoids the huge LLM cost of triple extraction,
# which is the right trade-off for a small single-document corpus.
DOMAIN_TERMS = [
    "collagen", "chitosan", "alginate", "hydrogel", "scaffold", "fibroin",
    "myoblast", "myotube", "differentiation", "cardiomyocyte", "elastomer",
    "biocompatib", "cytotoxic", "porosity", "stiffness", "modulus", "elasticity",
    "crosslink", "degradation", "fibre", "fiber", "electrospin", "alignment",
    "extracellular", "matrix", "osteoconduct", "osteoinduct", "hydroxyapatite",
    "calcium phosphate", "osseointegration", "bone", "implant", "inflammation",
    "macrophage", "fibrous capsule", "foreign body", "regeneration", "vascular",
    "contractile", "tissue engineering", "stem cell", "gelatin", "fibrin",
    "plga", "pcl", "peg", "decellulari", "mechanical", "strain", "corrosion",
    "wear", "fatigue", "stress shielding", "drug delivery", "swelling",
]


def _terms(text):
    t = text.lower()
    return {term for term in DOMAIN_TERMS if term in t}


def build(texts, min_shared=2):
    chunk_terms = {i: _terms(t) for i, t in enumerate(texts)}
    term_index = defaultdict(list)
    for i, terms in chunk_terms.items():
        for term in terms:
            term_index[term].append(i)

    neighbours = defaultdict(set)
    for idxs in term_index.values():
        for a in idxs:
            for b in idxs:
                if a != b and len(chunk_terms[a] & chunk_terms[b]) >= min_shared:
                    neighbours[a].add(b)
    return neighbours


def expand(seed_idxs, neighbours, hops=1, max_expand=15):
    expanded = set(seed_idxs)
    frontier = set(seed_idxs)
    for _ in range(hops):
        new = set()
        for i in frontier:
            new.update(neighbours.get(i, set()))
        new -= expanded
        expanded.update(new)
        frontier = new
        if len(expanded) >= len(seed_idxs) + max_expand:
            break
    return list(expanded)
