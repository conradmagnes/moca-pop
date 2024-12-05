"""!
    String Utilities.

"""


def get_section_break_str(
    text: str, filler_char: str = "=", total_length: int = 20
) -> str:
    """!Get a section break string."""
    filler_len = len(filler_char)
    num_fillers = (total_length - len(text)) // filler_len
    num_left = num_fillers // 2
    num_right = num_fillers - num_left
    return f"\n{filler_char * num_left} {text} {filler_char * num_right}\n"
