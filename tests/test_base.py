import unittest

from cross_project.base import CrossProjectAnalysis
from cross_project.util import FileManager


class TestCrossProjectAnalysis(unittest.TestCase):
    def test__init(self):
        cpa = CrossProjectAnalysis(FileManager.path("tests","data", "test_base1.csv"))
        expected_output = {
            ("project1", "version1"): [
                {"Smell1": "True", "Smell2": "False", "Smell3": "True"},
                {"Smell1": "True", "Smell2": "False", "Smell3": "False"}],
            ("project1", "version2"): [
                {"Smell1": "True", "Smell2": "False", "Smell3": "True"},
            ],
            ("project2", "version1"): [
                {"Smell1": "False", "Smell2": "False", "Smell3": "True"},
            ],
            ("project2", "version2"): [
                {"Smell1": "True", "Smell2": "True", "Smell3": "True"},
            ],
            ("project3", "version1"): [
                {"Smell1": "True", "Smell2": "False", "Smell3": "False"},
            ]}
        actual_output = cpa._datasets
        self.assertDictEqual(actual_output, expected_output)