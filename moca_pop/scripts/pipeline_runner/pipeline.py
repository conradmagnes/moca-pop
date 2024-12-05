"""!
    Pipeline classes for running Vicon Nexus pipelines
    ================================================

    This module contains classes for configuring and running Vicon Nexus pipelines.

    In addition to the base attributes required by the Vicon Nexus SDK, the Pipeline class
    allows for the specification of gates that must be passed before running the pipeline.

    Gap Fill Pipelines are a subclass of Pipeline that include additional gates for checking
    the quality of the marker trajectories before running the pipeline.

    Example pipeline series can be found in the `config_builder` directory.
    The `vicon_default_example.py` file provides an example of a pipeline series using default Vicon Nexus pipelines.

    @author C. McCarthy
"""

import operator
from typing import Any, Literal, Optional, Union
import time
import logging

from pydantic import BaseModel, Field

from moca_pop.utils import vicon_utils, quality_check as qc

LOGGER = logging.getLogger("Pipeline")

OPERATOR_MAP = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "lt": operator.lt,
    "ge": operator.ge,
    "le": operator.le,
}

AGG_OPERATOR_MAP = {
    "all": all,
    "any": any,
    "none": lambda x: not any(x),
}


class AttributeGate(BaseModel):
    attribute: str = Field(description="Name of the attribute to check")
    ref_value: Any = Field(description="Reference value to check against")
    operator: Literal["eq", "ne", "gt", "lt", "ge", "le"] = Field(
        description="Operator to use for comparison. Accepts 'eq', 'ne', 'gt', 'lt', 'ge', 'le'"
    )

    def check(self, values: dict) -> bool:
        """!Check if the attribute stored in 'values' passes the gate.

        @param values Values to check against the gate.
        @return True if the attribute passes the gate, False otherwise
        """
        try:
            return OPERATOR_MAP[self.operator](values[self.attribute], self.ref_value)
        except KeyError as e:
            if self.operator not in OPERATOR_MAP:
                raise ValueError(f"Operator {self.operator} not supported.")
            if self.attribute not in values:
                raise ValueError(f"Attribute {self.attribute} not found in values.")
            raise e


class PipelineArgs(BaseModel):
    location: str = Field(
        default="",
        description="Location of the pipeline (e.g. Shared, Private, System). If blank, uses default search mechanism.",
    )
    timeout: int = Field(
        default=200,
        description="Time in seconds to wait for the pipeline to complete before timing out. Does not necessary end pipeline. See Vicon Nexus SDK documentation for more information.",
    )


class Pipeline(BaseModel):
    type: Literal["Pipeline"] = Field(default="Pipeline")
    name: str = Field(description="Name of the pipeline")
    args: PipelineArgs = Field(
        description="Configuration for the pipeline", default_factory=PipelineArgs
    )
    gates: Optional[list[AttributeGate]] = Field(
        default=None, description="Gates to check before running the pipeline"
    )
    gate_operator: Literal["all", "any", "none"] = Field(
        default="all",
        description="Operator to use for combining the gates. Accepts 'all', 'any', 'none'",
    )

    def run(self, vicon, gate_checks=None, *args, **kwargs) -> bool:
        """!Run the pipeline.

        @param vicon: Vicon Nexus SDK object
        @param attributes: Attributes to check against the gates
        @return False if the pipeline was skipped, True otherwise
        """
        if self.gates:
            check = [gate.check(gate_checks) for gate in self.gates]
            if not AGG_OPERATOR_MAP[self.gate_operator](check):
                LOGGER.info(f"Pipeline {self.name} skipped.")
                return False

        start = time.time()
        vicon.RunPipeline(self.name, self.args.location, self.args.timeout)
        end = time.time()

        LOGGER.info(f"Pipeline {self.name} completed in {end - start:.2f} seconds.")
        return True


class GapFillPipeline(Pipeline):
    type: Literal["GapFillPipeline"] = Field(default="GapFillPipeline")

    def generate_attributes(self, vicon, subject_name: str, log_quality: bool = True):
        """!Generate gap fill pipeline attributes related to the quality of the marker trajectories.

        @param vicon Vicon Nexus SDK object
        @param subject_name Name of the subject
        @param log_quality Whether to log the quality of the marker trajectories
        @return Dictionary of attributes, including the maximum number of gaps, total number of gaps,\ 
            minimum percentage labeled, and total percentage labeled
        """
        marker_trajectories = vicon_utils.get_marker_trajectories(vicon, subject_name)
        gaps = qc.get_num_gaps(marker_trajectories)
        labeled = qc.get_perc_labeled(marker_trajectories)

        if log_quality:
            qc.log_gaps(gaps)
            qc.log_labeled(labeled)

        total_labeled = (
            sum(labeled.values()) / len(labeled) if len(labeled) > 0 else 100
        )
        min_labeled = min(labeled.values()) if len(labeled) > 0 else 100
        attributes = {
            "max_gaps": max(gaps.values()),
            "total_gaps": sum(gaps.values()),
            "min_perc_labeled": min_labeled,
            "total_perc_labeled": total_labeled,
        }

        return attributes

    def run(
        self,
        vicon,
        gate_checks=None,
        subject_name: str = None,
        log_quality: bool = True,
        *args,
        **kwargs,
    ) -> bool:
        """!Run the gap fill pipeline. If a subject name is provided, generate additional attributes based on the quality of the marker trajectories.
        The quality of the marker trajectories can be logged.

        @param vicon Vicon Nexus SDK object
        @param gate_checks Values to check against the gates. Set to None to generate values based on the quality of the marker trajectories
        @param subject_name Name of the subject (optional).
        @param log_quality Whether to log the quality of the marker trajectories
        """

        check_values = {}

        if gate_checks:
            check_values.update(gate_checks)

        if subject_name:
            check_values.update(
                self.generate_attributes(vicon, subject_name, log_quality)
            )

        if self.gates:
            check = [gate.check(check_values) for gate in self.gates]
            if not AGG_OPERATOR_MAP[self.gate_operator](check):
                LOGGER.info(f"Pipeline {self.name} skipped.")
                return False

        start = time.time()
        vicon.RunPipeline(self.name, self.args.location, self.args.timeout)
        end = time.time()

        LOGGER.info(f"Pipeline {self.name} completed in {end - start:.2f} seconds.")
        return True


class PipelineSeries(BaseModel):
    type: Literal["PipelineSeries"] = Field(default="PipelineSeries")
    pipelines: list[Union[Pipeline, GapFillPipeline, "PipelineSeries"]] = Field(
        description="Series of pipelines to run"
    )
    break_on_skip: bool = Field(description="Break the series if a pipeline is skipped")

    def run(self, vicon, gate_checks=None, *args, **kwargs) -> bool:
        """!Run the pipeline series.

        @param vicon Vicon Nexus SDK object
        @param gate_checks Values used to check against the gates
        @return True
        """
        for pipeline in self.pipelines:
            ran = pipeline.run(vicon, gate_checks=gate_checks, *args, **kwargs)
            if not ran and self.break_on_skip:
                break

        return True
