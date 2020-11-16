import pytest

from cross_project.base import CrossProjectAnalysis
from cross_project.models import Strategy
from cross_project.util import FileManager


class TestStrategy:
    def test_single_project_strategy_wrong_method(self):
        cpa = CrossProjectAnalysis(FileManager.path("data", "datasets.csv"))
        datasets = cpa._datasets
        with pytest.raises(ValueError):
            Strategy.single_project_strategy(datasets, "camel", "systemml", None, {}, None, {}, "wrong")

    def test_single_project_strategy_all(self):
        cpa = CrossProjectAnalysis(FileManager.path("data", "datasets.csv"))
        datasets = cpa._datasets
        with pytest.raises(AttributeError):
            Strategy.single_project_strategy(datasets, "camel", "systemml", None, {}, None, {}, "all")

    def test_single_project_strategy_separate(self):
        cpa = CrossProjectAnalysis(FileManager.path("data", "datasets.csv"))
        datasets = cpa._datasets
        with pytest.raises(AttributeError):
            Strategy.single_project_strategy(datasets, "camel", "systemml", None, {}, None, {}, "separate")
