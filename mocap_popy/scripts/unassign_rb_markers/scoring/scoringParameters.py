"""!
    Scoring Validator
    =================

    This module is used to generate or validate scoring parameters.

    @author C. McCarthy
"""

from typing import Literal, Union

from pydantic import BaseModel, Field


class PreAggTresholding(BaseModel):
    """!Whether to apply thresholding to residuals before aggregation."""

    segment: bool = Field(
        True, description="Pre-aggregation thresholding for segment residuals."
    )
    joint: bool = Field(
        True, description="Pre-aggregation thresholding for joint residuals."
    )


class AggregationMethod(BaseModel):
    """!Aggregation method for each component type (either 'sum' or 'mean'). Applied at each node."""

    segment: Literal["mean", "sum"] = Field(
        "mean", description="Aggregation method for segment residuals."
    )
    joint: Literal["mean", "sum"] = Field(
        "mean", description="Aggregation method for joint residuals."
    )


class AggregationWeight(BaseModel):
    """!Weight for each component type when adding together (aggregated) component residuals for each node."""

    segment: float = Field(
        1.0, description="Weight for segment (aggregated) residuals."
    )
    joint: float = Field(
        0.00278, description="Weight for joint (aggregated) residuals."
    )


class ScoringParameters(BaseModel):
    preagg_thresholding: PreAggTresholding = Field(
        description="Pre-aggregation thresholding for residuals.",
        default_factory=PreAggTresholding,
    )
    aggregation_method: AggregationMethod = Field(
        description="Aggregation method for residuals.",
        default_factory=AggregationMethod,
    )
    aggregation_weight: AggregationWeight = Field(
        description="Weight for aggregated residuals.",
        default_factory=AggregationWeight,
    )
    removal_threshold: Union[float, dict[str, float]] = Field(
        0.0, description="Threshold for removing markers from rigid body."
    )


if __name__ == "__main__":
    sp = ScoringParameters()
    print(sp.model_dump_json())
