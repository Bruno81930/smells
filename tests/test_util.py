import unittest
import os

from cross_project.util import FileManager

class TestFileManager(unittest.TestCase):
    def test_path(self):
        actual_path = FileManager.path("data", "dataset.csv")
        expected_path = FileManager.base + os.path.sep + "data" + os.path.sep + "dataset.csv"
        self.assertEqual(actual_path, expected_path)

if __name__ == "__main__":
    unittest.main()