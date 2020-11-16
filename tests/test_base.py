from cross_project.base import CrossProjectAnalysis
from cross_project.util import FileManager
from cross_project.models import Model
from errors import WrongMetricValue
import pytest

import pandas as pd


class TestCrossProjectAnalysis:
    def test__init(self):
        cpa = CrossProjectAnalysis(FileManager.path("tests", "data", "test_base1.csv"))
        expected_output = {
            ("project1", "version1"):
                pd.DataFrame([{"Smell1": "True", "Smell2": "False", "Smell3": "True"},
                              {"Smell1": "True", "Smell2": "False", "Smell3": "False"}]),
            ("project1", "version2"):
                pd.DataFrame([{"Smell1": "True", "Smell2": "False", "Smell3": "True"}]),
            ("project2", "version1"):
                pd.DataFrame([{"Smell1": "False", "Smell2": "False", "Smell3": "True"}]),
            ("project2", "version2"):
                pd.DataFrame([{"Smell1": "True", "Smell2": "True", "Smell3": "True"}]),
            ("project3", "version1"):
                pd.DataFrame([{"Smell1": "True", "Smell2": "False", "Smell3": "False"}])
            }
        actual_output = cpa._datasets

    def test_model_normalization(self):
        cpa = CrossProjectAnalysis(FileManager.path("data", "datasets.csv"))
        cpa.evaluate(Model.normalization, {"clf": {}, "clf_args": {}}, metrics={}, path="")
