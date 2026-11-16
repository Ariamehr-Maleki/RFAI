import unittest
from audit_candidate_contribution import compare


class CandidateContributionTests(unittest.TestCase):
    def test_retrieval_exchange_and_relevance_subsets(self):
        common = dict(qid='q', seed='s', date=1, truth=['new', 'repeat'],
                      new_truth=['new'], seed_history_count=3, training_positive_injections=0)
        padding = [str(i) for i in range(999)]
        full = dict(common, candidates=padding + ['new'])
        other = dict(common, candidates=padding + ['repeat'])
        value = compare(full, other)
        self.assertEqual(value['candidate_overlap'], 999)
        self.assertEqual(value['metrics']['all']['difference'], 0)
        self.assertEqual(value['metrics']['new']['difference'], 1)
        self.assertEqual(value['metrics']['repeat']['difference'], -1)
        self.assertEqual(value['metrics']['all']['relevant_gained'], 1)
        self.assertEqual(value['metrics']['all']['relevant_lost'], 1)
        no_new = dict(common, new_truth=[])
        self.assertNotIn('new', compare(dict(no_new, candidates=full['candidates']),
                                       dict(no_new, candidates=other['candidates']))['metrics'])
        with self.assertRaises(ValueError):
            compare(full, dict(other, truth=['different']))
        with self.assertRaises(ValueError):
            compare(dict(full, training_positive_injections=1), other)


if __name__ == '__main__':
    unittest.main()
