from toolz import thread_last
from functools import partial

from .util import read_rows, list_to_dict_two_levels

class CrossProjectAnalysis:
    def __init__(self, path):
        self._datasets = self._init(path)
        self.path = path

    def _init(self, path):
        transform2dict = partial(list_to_dict_two_levels, level1="Project", level2="Version")
        return thread_last(
            path, 
            read_rows,
            transform2dict
            )


    
