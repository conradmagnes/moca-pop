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

def run_in_new_console(close_console_on_exit: bool = False, new_console_flag:str = "--_new_console_opened"):
    """!Call the script in new terminal / console instance.

    The script must have a flag to avoid infinited loop of running script in new console.
    Default is '--_new_console_opened'.

    @param close_console_on_exit Whether to close the terminal/console after the script finishes (i.e. on exit).
    @param new_console_flag Script flag to handle call in newly opened terminal.
    """
    system = platform.system()
    
    if system == "Windows":
        close_flag = "/c" if close_console_on_exit else "/K"
        subprocess.run(['start', 'cmd.exe', close_flag, 'python', *sys.argv, new_console_flag], shell=True)
        exit(0)

    LOGGER.info(f"Unsupported platform for opening new console: {system}")
    exit(-1)