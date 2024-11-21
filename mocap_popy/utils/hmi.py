"""
    Human Machine Interface utilities
    ================================

    @author C. McCarthy
"""

import enum
from typing import Union
import logging


LOGGER = logging.getLogger("HMI")

QUIT_KEYWORDS = ("q", "quit", "exit")
YES_KEYWORDS = ("y", "yes")
NO_KEYWORDS = ("n", "no")


class ANSI_COLOR(enum.Enum):
    """!ANSI color codes"""

    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37


def color_string(
    string: str, color: Union[int, ANSI_COLOR], bright: bool = False
) -> str:
    """!Format string to be colored

    @param string String to be colored
    @param color Color to be applied to string
    @param bright Wheteher to use bright color. Default False
    @return Colorized string
    """
    offset = 60 if bright else 0
    c = color.value if isinstance(color, ANSI_COLOR) else color
    return f"\033[{c + offset:d}m{string}\033[m"


def get_user_input(
    prompt: str,
    exit_on_quit: bool = False,
    choices: list[tuple] = None,
    num_tries: int = 10,
    color_kwargs: dict = None,
) -> str:
    """!Colored input wrapper

    @param prompt Input prompt
    @param exit_on_quit Whether to exit on quit keyword
    @param choices List of choices to check against. Choices should be tuples of accepted inputs.
                If empty, any input is accepted and returned. Otherwise, index of choice is returned.
    @param num_tries Number of tries before returning -1.
    @param color_kwargs Keyword arguments for color_string

    """
    global LOGGER, QUIT_KEYWORDS
    default_colkw = {"color": ANSI_COLOR.BLUE, "bright": True}
    if color_kwargs is not None:
        default_colkw.update(color_kwargs)

    choices = choices or []

    i = 0
    retry = len(choices) > 0

    while i < num_tries:
        ui = input(color_string(f"-> {prompt}", **default_colkw))
        if ui.lower() in QUIT_KEYWORDS:
            LOGGER.info("Exit key pressed. Exiting.")
            if exit_on_quit:
                exit(0)
            return -1

        for idx, c in enumerate(choices):
            if ui.lower() in c:
                return idx

        if not retry:
            break

        LOGGER.warning(f"Invalid input. Please try again.")
        i += 1

    return -1 if retry else ui
