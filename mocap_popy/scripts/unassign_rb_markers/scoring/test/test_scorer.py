import unittest

import numpy as np


import mocap_popy.scripts.unassign_rb_markers.scoring.scorer as scorer

class TestRigidBodyScorer(unittest.TestCase):

    def test_sort_marker_scores(self):
        scores = {
            "marker1": 0.1,
            "marker2": 0.2,
            "marker3": 0.3,
        }
        sorted_scores = scorer.sort_marker_scores(scores)
        self.assertTrue(all([ss in [(m, s) for m, s in scores.items()] for ss in sorted_scores]))

        equal_scores = {
            "marker1": 0.1,
            "marker2": 0.1,
            "marker3": 0.1,
        }
        sorted_scores_incl = scorer.sort_marker_scores(equal_scores, max_markers=2, include_duplicates=True)
        self.assertTrue(all([ss in [(m, s) for m, s in equal_scores.items()] for ss in sorted_scores_incl]))
        
        sorted_scores_excl = scorer.sort_marker_scores(equal_scores, max_markers=2, include_duplicates=False)
        self.assertEqual(len(sorted_scores_excl), 0)

        mixed_scores = {
            "marker1": 0.1,
            "marker2": 0.2,
            "marker3": 0.1,
        }
        sorted_scores_mixed_incl = scorer.sort_marker_scores(mixed_scores, max_markers=2, include_duplicates=True)
        self.assertTrue(all([ss in [(m, s) for m, s in mixed_scores.items()] for ss in sorted_scores_mixed_incl]))

        sorted_scores_mixed_excl = scorer.sort_marker_scores(mixed_scores, max_markers=2, include_duplicates=False)
        self.assertEqual(sorted_scores_mixed_excl, [("marker2", 0.2)])





if __name__ == "__main__":
    unittest.main()