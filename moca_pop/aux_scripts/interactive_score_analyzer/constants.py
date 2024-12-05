"""!
    Constants for Interactive Score Analyzer App
    ============================================

    @author C. McCarthy

"""

## MISC ##
BROSWER_MARGINS = 8


## COLORS ##
CALIBRATED_COLOR = "gray"
WITHIN_THRESHOLD_COLOR = "black"
EXCEED_THRESHOLD_COLOR = "red"


def get_component_color(value, threshold: float = 0):
    return EXCEED_THRESHOLD_COLOR if value > threshold else WITHIN_THRESHOLD_COLOR
