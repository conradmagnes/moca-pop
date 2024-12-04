"""!
    Class definitions for NUSHU pipeline series configuration.
    ==========================================================

    The series requires the following pipelines:
        - ETH_NUSHU_R&L
        - ETH_NUSHU_MocaPop_Swap
        - ETH_NUSHU_MocaPop_Unassign
        - ETH_NUSHU_DeleteUnlabeled
        - ETH_NUSHU_FillGaps_Pass1
        - ETH_NUSHU_FillGaps_Pass2
        - ETH_NUSHU_FillGaps_Pass3
        - ETH_NUSHU_Filter
        - ETH_NUSHU_Export

    The series consists of the following steps:
        1) Reconstruct and Label (ETH_NUSHU_R&L)
        2) Swap Rigid Body markers with MocaPop (ETH_NUSHU_MocaPop_Swap)
        3) Unassign Rigid Body Markers with MocaPop (ETH_NUSHU_UnlabelRB)
        4) Delete Unlabeled Trajectory Markers (ETH_NUSHU_DeleteUnlabeled)
        4) Gap Fill Series (skips steps if no gaps remained)
            i) Small gap fill with Woltering, Rigid Body, and Pattern Fill (ETH_NUSHU_FillGaps_Pass1) 
            ii) Medium to Large gap fill with Kinematic Gap Fill and Rigid Body Fill (ETH_NUSHU_FillGaps_Pass2)
            iii) Fill remaining gaps with Kinematic Gap Fill (ETH_NUSHU_FillGaps_Pass3)
        5) (Optional) Butterworth Filter (ETH_NUSHU_Filter)
        6) (Optional) Export to C3D (ETH_NUSHU_Export)

    @author C. McCarthy
"""

import os

from mocap_popy.config import directory
from mocap_popy.utils import json_utils
import mocap_popy.scripts.pipeline_runner.pipeline as pipeline


rl_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_R&L", args=pipeline.PipelineArgs(location="Shared")
)

swap_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_MocaPop_Swap",
    args=pipeline.PipelineArgs(location="Shared", timeout=300),
)

unassign_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_MocaPop_Unassign",
    args=pipeline.PipelineArgs(location="Shared", timeout=300),
)

delete_unlabeled_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_DeleteUnlabeled",
    args=pipeline.PipelineArgs(location="Shared", timeout=60),
)

fill_gap_pipelines = [
    pipeline.Pipeline(
        name=f"ETH_NUSHU_FillGaps_Pass{i}",
        args=pipeline.PipelineArgs(location="Shared", timeout=200),
        gates=[
            pipeline.AttributeGate(attribute="total_gaps", ref_value=0, operator="gt")
        ],
    )
    for i in range(1, 4)
]

fill_gap_series = pipeline.GapFillPipelineSeries(
    pipelines=fill_gap_pipelines, break_on_skip=True
)

filter_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_Filter",
    args=pipeline.PipelineArgs(location="Shared", timeout=60),
    gates=[pipeline.AttributeGate(attribute="filter", ref_value=True, operator="eq")],
)

export_pipeline = pipeline.Pipeline(
    name="ETH_NUSHU_Export",
    args=pipeline.PipelineArgs(location="Shared", timeout=100),
    gates=[pipeline.AttributeGate(attribute="export", ref_value=True, operator="eq")],
)

nushu_pipeline_series = pipeline.PipelineSeries(
    pipelines=[
        rl_pipeline,
        swap_pipeline,
        unassign_pipeline,
        delete_unlabeled_pipeline,
        fill_gap_series,
        filter_pipeline,
        export_pipeline,
    ],
    break_on_skip=False,
)


model_json = nushu_pipeline_series.model_dump_json(indent=4)
output_dir = os.path.join(directory.SCRIPTS_DIR, "nushu_pipeline_runner", "config")
output_fn = os.path.join(output_dir, "nushu_pipeline_series.json")

with open(output_fn, "w") as f:
    f.write(model_json)
