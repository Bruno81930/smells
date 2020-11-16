from functools import partial
from typing import Union, Callable, List, Dict

import pandas as pd
from toolz import thread_last

from .util import read_rows, list_to_dict_two_levels, apply_to_pandas_df
from errors import WrongMetricValue


class CrossProjectAnalysis:
    def __init__(self, path):
        self._datasets = self._init(path)
        self.path = path

    def _init(self, path):
        return thread_last(
            path,
            read_rows,
            list_to_dict_two_levels,
            apply_to_pandas_df
        )

    def evaluate(
            self,
            model: Callable,
            model_arguments: Dict,
            metrics: Dict,
            path: Union[str, bool]):

        results = model(self._datasets, metrics=metrics, **model_arguments)

        if path:
            pd.Dataframe(results).to_csv(path, index=False)

        return results
