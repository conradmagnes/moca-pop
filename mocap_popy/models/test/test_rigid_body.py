import unittest as ut

import numpy as np

from mocap_popy.models.rigid_body import Node, Segment, Joint


class TestRigidBody(ut.TestCase):
    def test_segment(self):
        n1 = Node("n1", np.array([0, 0, 0]), True)
        n2 = Node("n2", np.array([1, 0, 0]), True)
        n3 = Node("n3", np.array([1, 1, 1]), True)
        n3_mock = Node("n3", np.array([0, 0, 0]), False)
        s1 = Segment(n1, n2)
        s2 = Segment(n2, n3)
        s3 = Segment(n2, n3_mock)
        s4 = Segment(n2, n1)

        self.assertEqual(s1.get_nodes(), [n1, n2])
        self.assertEqual(s1.length, 0)

        s1.compute_length()
        self.assertEqual(s1.length, 1)

        s3.compute_length()
        self.assertEqual(s3.length, 0)

        self.assertTrue(s1.exists)
        self.assertFalse(s3.exists)

        self.assertTrue(s1.shares_nodes(s1))
        self.assertTrue(s1.shares_nodes((n1, n2)))
        self.assertFalse(s1.shares_nodes(s2))
        self.assertFalse(s1.shares_nodes((n1, n3)))
        self.assertTrue(s2.shares_nodes(s3))

        s2.compute_length()
        s4.compute_length()
        self.assertFalse(s2 == s3)
        self.assertTrue(s1 == s4)

        self.assertTrue(n1.marker in s1)
        self.assertFalse(n1.marker in s2)

    def test_joint(self):
        n1 = Node("n1", np.array([0, 0, 0]), 1.0)
        n2 = Node("n2", np.array([0, 0, 0]), 1.0)
        n3 = Node("n3", np.array([0, 0, 0]), 1.0)
        n4 = Node("n4", np.array([0, 0, 0]), 1.0)

        s1 = Segment(n1, n2)
        s2 = Segment(n2, n3)
        s3 = Segment(n1, n3)
        s4 = Segment(n2, n1)
        s5 = Segment(n2, n4)

        j1 = Joint(n2, s1, s2)
        j2 = Joint(n2, s2, s4)
        j3 = Joint(n1, s1, s3)
        j4 = Joint(n2, s4, s2)
        j5 = Joint(n2, s1, s5)

        self.assertEqual(j1.get_nodes(), [n1, n2, n3])
        self.assertEqual(j1.get_segments(), [s1, s2])
        self.assertTrue(j1.shares_segments(j1))
        self.assertTrue(j1.shares_segments((s1, s2)))
        self.assertTrue(j1.shares_segments(j2))
        self.assertTrue(j1.shares_segments(j3))
        self.assertFalse(j1.shares_segments(j5))
        self.assertTrue(j1 == j4)
        self.assertFalse(j1 == j3)

        self.assertTrue(j1 in [j1, j2])
        self.assertTrue(j1 in [j3, j4])
        self.assertFalse(j1 in [j3, j5])


if __name__ == "__main__":
    ut.main()
