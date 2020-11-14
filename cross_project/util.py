import os
from pathlib import Path
from csv import reader

class FileManager:
    base = os.getcwd() 
   
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


def list_to_dict_two_levels(dataset, level1, level2):
    def key(row):
        return (row[0],row[1])
    
    def process(row, columns):
        return dict(zip(columns, row[4:]))

    dataset_dict = dict()
    columns = dataset.pop(0)[4:]
    [dataset_dict.setdefault(key(row), []).append(process(row, columns)) 
            for row in dataset]
    return dataset_dict




