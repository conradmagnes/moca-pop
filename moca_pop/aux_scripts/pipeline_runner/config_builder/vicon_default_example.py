"""!
    Vicon default example configuration file.
    ========================================

    This file provides an example of a pipeline series using default Vicon Nexus pipelines.

    The pipeline series consists of two pipelines:
        - Reconstruct And Label
        - Auto Intelligent Gap Fill

    @author C. McCarthy
"""

import os

from moca_pop.config import directory
from moca_pop.utils import json_utils
import moca_pop.aux_scripts.pipeline_runner.pipeline as pipeline

rl_pipeline = pipeline.Pipeline(name="Reconstruct And Label")

gap_fill_pipeline = pipeline.GapFillPipeline(
    name="Auto Intelligent Gap Fill",
    gates=[pipeline.AttributeGate(attribute="total_gaps", ref_value=0, operator="gt")],
)

default_pipeline_series = pipeline.PipelineSeries(
    pipelines=[rl_pipeline, gap_fill_pipeline], break_on_skip=False
)

model_json = default_pipeline_series.model_dump_json(indent=4)
output_dir = os.path.join(directory.AUX_DIR, "pipeline_runner", "config")
output_fn = os.path.join(output_dir, "vicon_default_example.json")

with open(output_fn, "w") as f:
    f.write(model_json)

## test import
import_str = json_utils.import_json_as_str(output_fn)
default_pipeline_series_test = pipeline.PipelineSeries.model_validate_json(import_str)
