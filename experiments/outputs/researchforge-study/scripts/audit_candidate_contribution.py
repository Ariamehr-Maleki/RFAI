"""Paired descriptive DEVELOPMENT retrieval audit; no rankings or final labels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'
BASE = STUDY / 'results/prescience-expanded-1200-1000-500-1000'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(full, no_content):
    for key in ['qid', 'seed', 'date', 'truth', 'new_truth', 'seed_history_count']:
        if full[key] != no_content[key]:
            raise ValueError('Paired metadata differs: ' + key)
    for row in [full, no_content]:
        if row['training_positive_injections'] or len(set(row['candidates'])) != 1000:
            raise ValueError('Expected untouched 1000-candidate validation pool')
    a, b = set(full['candidates']), set(no_content['candidates'])
    truth, new = set(full['truth']), set(full['new_truth'])
    result = dict(qid=full['qid'], seed=full['seed'], history=full['seed_history_count'],
                  candidate_overlap=len(a & b), metrics={})
    for name, labels in [('all', truth), ('new', new), ('repeat', truth - new)]:
        if not labels:
            continue
        ra, rb = len(a & labels) / len(labels), len(b & labels) / len(labels)
        result['metrics'][name] = dict(full=ra, no_content=rb, difference=ra-rb,
            relevant_gained=len((a-b) & labels), relevant_lost=len((b-a) & labels))
    return result


def main():
    sources = {}
    for mode in ['full', 'no_content']:
        audit = STUDY / 'data-audit' / f'expanded-{mode}-cohort.json'
        if not read(audit)['status'].startswith('PASS'):
            raise ValueError('Cohort audit required')
        sources[str(audit.relative_to(ROOT))] = sha(audit)
    plan = BASE / 'query_manifest.json'
    sources[str(plan.relative_to(ROOT))] = sha(plan)
    groups = defaultdict(list)
    receipt_hash = hashlib.sha256()
    for query in read(plan)['validation']:
        pair = []
        for mode in ['full', 'no_content']:
            path = BASE / mode / 'valid_queries' / (str(query['qid']) + '.json')
            receipt = read(path.with_suffix('.receipt'))
            value = sha(path)
            if value != receipt['metadata_sha256']:
                raise ValueError('Committed metadata changed')
            receipt_hash.update((mode + ':' + path.name + ':' + value + '\n').encode())
            pair.append(read(path))
        row = compare(*pair)
        groups['all_queries'].append(row)
        h = row['history']
        groups['history_' + ('1-3' if h <= 3 else '4-10' if h <= 10 else '11+')].append(row)
    summary = {}
    for group, rows in groups.items():
        cell = dict(queries=len(rows), mean_candidate_overlap=statistics.mean(r['candidate_overlap'] for r in rows), relevance={})
        for relevance in ['all', 'new', 'repeat']:
            eligible = [r for r in rows if relevance in r['metrics']]
            values = [r['metrics'][relevance] for r in eligible]
            cell['relevance'][relevance] = dict(eligible_queries=len(values), distinct_authors=len({r['seed'] for r in eligible}),
                **{key: statistics.mean(v[key] for v in values) for key in ['full', 'no_content', 'difference']},
                queries_improved=sum(v['difference'] > 0 for v in values),
                queries_worsened=sum(v['difference'] < 0 for v in values),
                queries_tied=sum(v['difference'] == 0 for v in values),
                gained_query_label_occurrences=sum(v['relevant_gained'] for v in values),
                lost_query_label_occurrences=sum(v['relevant_lost'] for v in values))
        summary[group] = cell
    result = dict(status='Completed paired descriptive DEVELOPMENT candidate audit; no confirmatory inference',
        source_sha256=sources, paired_metadata_sha256=receipt_hash.hexdigest(), audit_code_sha256=sha(Path(__file__)),
        summary=summary, limitations=[
            'Macro recall averages only queries with a nonempty relevant subset; new and repeat groups overlap.',
            'Gained/lost counts are query-label occurrences, not distinct people.',
            'The full retrieval pipeline adds content channels and changes fusion; these differences do not isolate semantic versus lexical causality.',
            'Adaptive validation data; no ranking superiority, significance or final-test claim.'])
    atomic_state(STUDY / 'data-audit/expanded-candidate-contribution.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
