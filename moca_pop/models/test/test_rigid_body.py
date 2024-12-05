import unittest as ut

import numpy as np

from moca_pop.models.rigid_body import Node, Segment, Joint, RigidBody


class TestRigidBody(ut.TestCase):

    def test_node(self):
        n1 = Node("n1", np.array([0, 0, 0]), True)
        n2 = Node("n2", np.array([1, 0, 0]), True)
        n3 = Node("n1", np.array([1, 1, 1]), False)

        self.assertEqual(n1.marker, "n1")
        self.assertTrue(n1 in [n1, n2])
        self.assertTrue(n1 in [n2, n3])
        self.assertFalse(n2 in [n1, n3])

    def test_segment(self):
        n1 = Node("n1", np.array([0, 0, 0]), True)
        n2 = Node("n2", np.array([1, 0, 0]), True)
        n3 = Node("n3", np.array([1, 1, 1]), True)
        n3_mock = Node("n3", np.array([0, 0, 0]), False)
        s1 = Segment((n1, n2))
        s2 = Segment((n2, n3))
        s3 = Segment((n2, n3_mock))
        s4 = Segment((n2, n1))

        self.assertEqual(s1.nodes, (n1, n2))
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

    def test_determine_shared_node(self):
        s1 = Segment(("n1", "n2"))
        s2 = Segment(("n2", "n3"))
        s3 = Segment(("n1", "n3"))

        self.assertEqual(Joint.determine_shared_node((s1, s2)).marker, "n2")

    def test_joint(self):
        n1 = Node("n1", np.array([0, 0, 0]), 1.0)
        n2 = Node("n2", np.array([0, 0, 0]), 1.0)
        n3 = Node("n3", np.array([0, 0, 0]), 1.0)
        n4 = Node("n4", np.array([0, 0, 0]), 1.0)

        s1 = Segment((n1, n2))  # "n1-n2"
        s2 = Segment((n2, n3))  # "n2-n3"
        s3 = Segment((n1, n3))  # "n1-n3"
        s4 = Segment((n2, n1))  # "n2-n1"
        s5 = Segment((n2, n4))  # "n2-n4"

        j1 = Joint((s1, s2))  # "n1-n2-n3"
        j2 = Joint((s2, s4))  # "n2-n3-n1"
        j3 = Joint((s1, s3))  # "n2-n1-n3"
        j4 = Joint((s4, s2))  # "n1-n2-n3"
        j5 = Joint((s1, s5))  # "n1-n2-n4"

        self.assertEqual(j1.segments, (s1, s2))
        self.assertEqual(j1.get_nodes(), [n1, n2, n3])
        self.assertTrue(j1.shares_segments(j1))
        self.assertTrue(j1.shares_segments((s1, s2)))
        self.assertTrue(j1.shares_segments((n1, n2, n3)))
        self.assertTrue(j1.shares_segments(("n1", n2, "n3")))
        self.assertTrue(j1.shares_segments("n1-n2-n3"))
        self.assertTrue(j1.shares_segments(j2))
        self.assertFalse(j1.shares_segments(j3))
        self.assertFalse(j1.shares_segments(j5))
        self.assertTrue(j1 == j4)
        self.assertFalse(j1 == j3)

        self.assertTrue(j1 in [j1, j2])
        self.assertTrue(j1 in [j3, j4])
        self.assertFalse(j1 in [j3, j5])

    def test_rigid_body(self):
        n1 = Node("n1", np.zeros(3), True)
        n2 = Node("n2", np.zeros(3), False)
        n3 = Node("n3", np.zeros(3), True)
        s1 = Segment((n1, n2))
        s2 = Segment((n2, n3))
        j1 = Joint((s1, s2))

        body = RigidBody("body", [n1, n2, n3], [s1, s2])

        self.assertEqual(body.name, "body")
        self.assertEqual(body.nodes, [n1, n2, n3])
        self.assertEqual(body.segments, [s1, s2])
        self.assertEqual(body.joints, [j1])

        self.assertEqual(body.get_node("n1"), n1)
        self.assertEqual(body.get_segment("n1-n2"), s1)
        self.assertEqual(body.get_segment((n1, n2)), s1)

        self.assertEqual(body.get_joint("n1-n2-n3"), j1)
        self.assertNotEqual(body.get_joint((n2, n1, n3)), j1)
        self.assertEqual(body.get_joint((s1, s2)), j1)

        self.assertEqual(body.get_joint(("n1-n2", s2)), j1)
        self.assertIsNone(body.get_joint(("n1-n2", "n1-n3")))

        # duplicate nodes
        with self.assertRaises(ValueError):
            body2 = RigidBody("body", [n1, n1, n2], [s1, s2])

    def test_rigid_body_residuals(self):
        n1 = Node("n1", np.array([1, 0, 0]), True)
        n2 = Node("n2", np.array([0, 0, 0]), True)
        n3 = Node("n3", np.array([0, 1, 0]), True)
        rb1 = RigidBody(
            "rb1",
            nodes=[n1, n2, n3],
            segments=["n1-n2", "n2-n3"],
            compute_segment_lengths=True,
            compute_joint_angles=True,
        )

        n1_2 = Node("n1", np.array([2, 0, 0]), True)
        n1_3 = Node("n3", np.array([1, 1, 0]), True)

        rb2 = RigidBody(
            "rb2",
            nodes=[n1_2, n2, n1_3],
            segments=rb1.segments,
            compute_joint_angles=True,
            compute_segment_lengths=True,
        )

        self.assertAlmostEqual(rb1.get_segment("n1-n2").length, 1)
        self.assertAlmostEqual(rb2.get_segment("n1-n2").length, 2)

        self.assertAlmostEqual(rb1.get_joint("n1-n2-n3").angle, 90)
        self.assertAlmostEqual(rb2.get_joint("n1-n2-n3").angle, 45)

        rb2.compute_segment_residuals(rb1)
        rb2.compute_joint_residuals(rb1)

        self.assertAlmostEqual(rb2.get_segment("n1-n2").residual_calib, 1)
        self.assertAlmostEqual(rb2.get_segment("n1-n2").residual_prior, 0)
        self.assertAlmostEqual(abs(rb2.get_joint("n1-n2-n3").residual_calib), 45)

        agg_n1_seg_calib, _ = rb2.get_aggregate_segment_residuals("n1")
        self.assertTrue(agg_n1_seg_calib == 1)
        agg_n2_seg_calib, _ = rb2.get_aggregate_segment_residuals("n2")
        self.assertTrue(agg_n2_seg_calib > 1)
        agg_n3_seg_calib, _ = rb2.get_aggregate_segment_residuals("n3")
        self.assertTrue(0 < agg_n3_seg_calib < 1)

        agg_n2_joint_calib, _ = rb2.get_aggregate_joint_residuals("n2")
        self.assertTrue(agg_n2_joint_calib > 0)

    def test_str_repr(self):
        n1 = Node("m1", np.zeros(3), True)
        n2 = Node("m2", np.zeros(3), False)
        n3 = Node("m3", np.zeros(3), True)
        s1 = Segment((n1, n2))
        s2 = Segment((n2, n3))
        j1 = Joint((s1, s2))

        body = RigidBody("body", [n1, n2, n3], [s1, s2])

        print(n1)
        print(repr(n1))
        print(s1)
        print(repr(s1))
        print(j1)
        print(repr(j1))
        print(body)
        print(repr(body))


if __name__ == "__main__":
    # trb = TestRigidBody()
    # trb.test_rigid_body_residuals()
    ut.main()
