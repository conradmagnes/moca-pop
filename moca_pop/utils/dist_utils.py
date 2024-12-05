"""!
    Distribution Utilities
    ======================

    Helpful Functions for distributing scripts to other platforms / software.

    @author C. McCarthy

"""

import logging
import platform
import subprocess
import sys

LOGGER = logging.getLogger("DistUtils")


def run_python_script(
    script_path: str,
    script_args: list[str],
    verbose: bool = False,
    python_path: str = "python",
) -> subprocess.CompletedProcess:
    """Run a script as a subprocess and wait for completion.

    @param script_path Path to the script to run.
    @param script_args List of arguments to pass to the script.
    @param verbose Whether to print the script output in real-time.
    @param python_path Path to the python executable. (default: 'python')

    @return Result of the subprocess execution.
    """
    command = [python_path, script_path, *script_args]
    LOGGER.debug(f"Running command: {' '.join(command)}")

    if verbose:
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        LOGGER.info(f"Script output:\n{process.stdout}")
        if process.returncode != 0:
            LOGGER.error(f"Script error:\n{process.stderr}")
    else:
        process = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    return process


def run_in_new_console(
    script_name: str = None,
    close_console_on_exit: bool = False,
    new_console_flag: str = "--_new_console_opened",
):
    """!Call the script in new terminal / console instance.

    The script must have a flag to avoid infinited loop of running script in new console.
    Default is '--_new_console_opened'.

    @param script_name Name of the script to run in new console. If None, uses the current script.
    @param close_console_on_exit Whether to close the terminal/console after the script finishes (i.e. on exit).
    @param new_console_flag Script flag to handle call in newly opened terminal.
    """
    system = platform.system()

    if system == "Windows":
        close_flag = "/c" if close_console_on_exit else "/K"
        script_name = script_name if script_name else sys.argv[0]
        subprocess.run(
            [
                "start",
                "cmd.exe",
                close_flag,
                "python",
                script_name,
                *sys.argv[1:],
                new_console_flag,
            ],
            shell=True,
        )
        exit(0)

    LOGGER.info(f"Unsupported platform for opening new console: {system}")
    exit(-1)
