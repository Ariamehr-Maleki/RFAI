"""Numerical port of original weights; explicit history-only benchmark adapter.

Original source commit57243e190904f6ca7e6796734c32878ba8cd6c45,
agent.py:calculate_advanced_match. No fabricated skills, citations or location.
The benchmark adapter cannot reproduce the original live retrieval/UI pipeline.
"""
import bisect


def original_score(researcher_skills=(), project_skills=(), researcher_topics=(),
                   project_topics=(), collaboration_type=None, semantic_cosine=None):
    rs = {s.lower() for s in researcher_skills}
    ps = {s.lower() for s in project_skills}
    skill = 100 * len(rs & ps) / len(rs | ps) if rs | ps else 0.
    rt = [s.lower() for s in researcher_topics]
    pt = [s.lower() for s in project_topics]
    interest = 100 * sum(any(p in r or r in p for p in pt) for r in rt) / len(rt) if rt and pt else 0.
    complementary = min(100 * len(rs - ps) / max(1, len(rs)), 100)
    geography = 85. if collaboration_type in ['International', 'remote'] else 50.
    career = 85.
    semantic = 50. if semantic_cosine is None else max(0., 100 * semantic_cosine)
    return round(.25 * skill + .25 * interest + .15 * complementary +
                 .10 * geography + .10 * career + .15 * semantic, 1)


def historical_topics(author, cutoff, papers, histories, dates):
    end = bisect.bisect_left(dates.get(author, []), cutoff)
    ids = histories.get(author, [])[:end][-10:]
    return sorted({c for pid in ids for c in papers[pid]['categories']})


def history_adapter(source, candidates, cutoff, papers, histories, dates):
    seed_topics = historical_topics(source, cutoff, papers, histories, dates)
    return [original_score(researcher_topics=historical_topics(a, cutoff, papers, histories, dates),
                           project_topics=seed_topics) for a in candidates]


ADAPTATION = {
    'weights': [.25, .25, .15, .10, .10, .15],
    'topics': 'Unique arXiv categories from last10 strictly earlier papers; proxy for research interests.',
    'skills': 'Unavailable; empty for both sides, so skill/complementarity terms zero.',
    'geography': 'Unavailable; preserve original non-remote default50 for every candidate.',
    'career': 'Preserve original constant85; do not infer seniority.',
    'semantics': 'Preserve original missing-embedding fallback50 in this history-category adapter.',
    'rounding': 'Original one-decimal final score; deterministic author-ID tie order in benchmark.',
    'scope': 'Data-adapted scoring comparator on shared candidate pool; not exact end-to-end original system.',
}
