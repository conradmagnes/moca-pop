import unittest
from unittest.mock import Mock
from pydantic import ValidationError

from moca_pop.scripts.pipeline_runner.pipeline import (
    AttributeGate,
    Pipeline,
    PipelineSeries,
    GapFillPipelineSeries,
    PipelineArgs,
)


class TestAttributeGate(unittest.TestCase):
    def setUp(self):
        self.mock_attributes = Mock()
        self.mock_attributes.some_attribute = 10

    def test_check_valid_operator(self):
        gate = AttributeGate(attribute="some_attribute", ref_value=10, operator="eq")
        self.assertTrue(gate.check(self.mock_attributes))

        gate = AttributeGate(attribute="some_attribute", ref_value=5, operator="gt")
        self.assertTrue(gate.check(self.mock_attributes))

        gate = AttributeGate(attribute="some_attribute", ref_value=20, operator="lt")
        self.assertTrue(gate.check(self.mock_attributes))

    def test_check_invalid_operator(self):
        with self.assertRaises(ValueError):
            gate = AttributeGate(
                attribute="some_attribute", ref_value=10, operator="unsupported"
            )
            gate.check(self.mock_attributes)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_attributes = Mock()
        self.mock_attributes.some_attribute = 10
        self.args = PipelineArgs(location="test_location", timeout=300)
        self.pipeline = Pipeline(
            name="TestPipeline",
            args=self.args,
            gates=[
                AttributeGate(attribute="some_attribute", ref_value=10, operator="eq"),
                AttributeGate(attribute="some_attribute", ref_value=5, operator="gt"),
            ],
            gate_operator="all",
        )

    def test_pipeline_passes_gates(self):
        result = self.pipeline.run(Mock(), self.mock_attributes)
        self.assertTrue(result)

    def test_pipeline_fails_gates(self):
        self.pipeline.gates[1] = AttributeGate(
            attribute="some_attribute", ref_value=15, operator="gt"
        )
        result = self.pipeline.run(Mock(), self.mock_attributes)
        self.assertFalse(result)

    def test_invalid_gate_operator(self):
        with self.assertRaises(ValidationError):
            pipeline = Pipeline(
                name="InvalidOperatorPipeline",
                args=self.args,
                gates=[
                    AttributeGate(
                        attribute="some_attribute", ref_value=10, operator="eq"
                    )
                ],
                gate_operator="invalid_operator",
            )
            pipeline.run(Mock(), self.mock_attributes)


class TestPipelineSeries(unittest.TestCase):
    def setUp(self):
        self.mock_attributes = Mock()
        self.mock_attributes.some_attribute = 10
        self.args = PipelineArgs(location="test_location", timeout=300)

        self.pipeline_1 = Pipeline(
            name="Pipeline1",
            args=self.args,
            gates=[
                AttributeGate(attribute="some_attribute", ref_value=10, operator="eq")
            ],
            gate_operator="all",
        )

        self.pipeline_2 = Pipeline(
            name="Pipeline2",
            args=self.args,
            gates=[
                AttributeGate(attribute="some_attribute", ref_value=5, operator="gt")
            ],
            gate_operator="all",
        )

        self.pipeline_series = PipelineSeries(
            pipelines=[self.pipeline_1, self.pipeline_2], break_on_skip=False
        )

    def test_pipeline_series_all_pipelines_run(self):
        vicon_mock = Mock()
        self.pipeline_series.run(vicon_mock, self.mock_attributes)

        vicon_mock.RunPipeline.assert_called()  # Ensure pipelines were executed

    def test_pipeline_series_break_on_skip(self):
        self.pipeline_2.gates[0] = AttributeGate(
            attribute="some_attribute", ref_value=15, operator="gt"
        )
        self.pipeline_series.break_on_skip = True

        vicon_mock = Mock()
        self.pipeline_series.run(vicon_mock, self.mock_attributes)

        # Only the first pipeline should run since the second one fails and break_on_skip=True
        vicon_mock.RunPipeline.assert_called_once()

    def test_nested_pipeline_series(self):
        pipeline_series_2 = PipelineSeries(
            pipelines=[self.pipeline_1, self.pipeline_2], break_on_skip=False
        )

        pipeline_series = PipelineSeries(
            pipelines=[self.pipeline_series, pipeline_series_2], break_on_skip=False
        )

        vicon_mock = Mock()
        pipeline_series.run(vicon_mock, self.mock_attributes)

        # Ensure all pipelines were executed
        self.assertEqual(vicon_mock.RunPipeline.call_count, 4)


if __name__ == "__main__":
    unittest.main()
