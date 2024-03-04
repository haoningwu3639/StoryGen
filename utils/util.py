import sys
import copy
import inspect
import datetime
from typing import Dict
from os.path import *


def get_time_string() -> str:
    x = datetime.datetime.now()
    return f"{(x.year - 2000):02d}{x.month:02d}{x.day:02d}-{x.hour:02d}{x.minute:02d}{x.second:02d}"

def get_function_args() -> Dict:
    frame = sys._getframe(1)
    args, _, _, values = inspect.getargvalues(frame)
    args_dict = copy.deepcopy({arg: values[arg] for arg in args})

    return args_dict