import os
from pathlib import Path
from csv import reader
from typing import List, Dict

import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FileManager:
    base = str(Path(__file__).parent.parent)

    @staticmethod
    def path(*args):
        d = args[:-1]
        Path(FileManager.base, *d).mkdir(parents=True, exist_ok=True)
        return str(Path(FileManager.base, *args))


def read_rows(path):
    with open(path, 'r') as data_obj:
        csv_reader = reader(data_obj)
        dataset = list(csv_reader)
    return dataset


def list_to_dict_two_levels(dataset):
    def key(row):
        return (row[0], row[1])

    def process(row, columns):
        return dict(zip(columns, row[4:]))

    dataset_dict = dict()
    columns = dataset.pop(0)[4:]
    [dataset_dict.setdefault(key(row), []).append(process(row, columns))
     for row in dataset]
    return dataset_dict


def apply_to_pandas_df(datasets_dict: Dict):
    return {key: pd.DataFrame(datasets_dict[key]) for key in datasets_dict.keys()}
