import os
from abc import ABC, abstractmethod
from csv import DictReader
import json
from csv import DictReader
from enum import Enum, auto
from functools import partial, reduce
from itertools import product, chain
from operator import itemgetter
from pathlib import Path
from typing import Tuple, Dict, ClassVar, Union, List, Callable

import numpy as np
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from scipy.spatial.distance import cosine, jaccard, rogerstanimoto
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, brier_score_loss, fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator


class Project:
    def __init__(self, name, versions=None, data=None, bool_features=True):
        self.name = name
        self.versions = [] if versions is None else versions
        self.data = [] if data is None else data
        self.bool_features = bool_features

    def __repr__(self):
        return f'Project({self.name}, {self.versions}, {repr(self.data[0])}'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def append(self, version: str, file: Tuple[str, str], features: Dict, bugged: str):
        if version not in self.versions:
            self.versions.append(version)

        self.data.append({
            'version': version,
            'file': file,
            'features': features,
            'bugged': bugged
        })

    def _get_x(self, versions):
        """ Accessing the data from the project and extract its features. """
        x = list(map(lambda _: list(_.values()),
                     map(lambda _: _['features'],
                         filter(lambda _: _['version'] in versions, self.data))))
        return np.array([[v == "True" for v in sub] for sub in x], dtype=int) if self.bool_features else np.array(x,
                                                                                                                  dtype=float)

    def _get_y(self, versions):
        """ Accessing the data from the project and extract the bugged information."""
        y = list(map(lambda _: _['bugged'],
                     filter(lambda _: _['version'] in versions, self.data)))
        return np.array([v == "True" for v in y], dtype=int) if self.bool_features else np.array(y, dtype=float)

    def get_set(self, set_type="train", strategy="standard"):
        """ Get the sets used for machine learning: (X_train/X_test, y_train/y_test)"""
        if set_type not in ["train", "test"]:
            raise ValueError("Wrong type for the set: (\"train\", \"test\")")

        if strategy == "standard":
            versions = self.versions[:4] if set_type == "train" else self.versions[4]
        elif strategy == "all":
            versions = self.versions
        else:
            raise ValueError("Wrong strategy type")

        return self._get_x(versions), self._get_y(versions)

    def get_X(self, set_type="train", strategy="standard"):
        return self.get_set(set_type, strategy)[0]

    def get_y(self, set_type="train", strategy="standard"):
        return self.get_set(set_type, strategy)[1]


class ModelCache:
    META_FILENAME = "meta.json"

    def __init__(self,
                 project_name: str,
                 classifier: BaseEstimator,
                 classifier_config: Dict,
                 variation: str = "common",
                 base_path: str = Path(Path(__file__).parent.parent, "data", "cache"),
                 ):
        self.project_name = project_name
        assert isinstance(classifier, BaseEstimator), "The Classifier needs to be an instance from sklearn"
        self.classifier = classifier
        self.classifier_config = classifier_config
        self.variation = variation
        self.key = self._create_key()
        self.base_path = base_path
        self.meta_path = str(Path(str(base_path), ModelCache.META_FILENAME))
        self.meta = None
        self.cache_path: str = ""
        self.cached = False

    def __enter__(self):
        self.meta = self._load_meta()
        if self.key in self.meta:
            self.cache_path = self.meta[self.key]
            self.classifier = load(self.cache_path)
            self.cached = True
        else:
            self.cache_path = self._create_path()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type and not self.cached:
            dump(self.classifier, self.cache_path)
            self.meta[self.key] = self.cache_path
            self._dump_meta()
            return False
        return True

    def fit(self, X, y):
        if not self.cached:
            if len(self.classifier_config) > 0:
                self.classifier.set_params(**self.classifier_config)
            self.classifier.fit(X, y)
        return self.classifier

    def _load_meta(self):
        try:
            with open(self.meta_path, 'r') as json_file:
                return json.load(json_file)
        except EnvironmentError:
            print(f'Failed to load Model Cache File: {self.meta_path}')
            return dict()
        except json.JSONDecodeError:
            print("Failed to load Model Cache from JSON")
            return dict()

    def _dump_meta(self):
        try:
            with open(self.meta_path, 'w') as json_file:
                return json.dump(self.meta, json_file)
        except EnvironmentError:
            print(f'Failed to dump Model to Cache File: {self.meta_path}')
        except json.JSONDecodeError:
            print("Failed to dump Model Cache from JSON")
        finally:
            return

    def _create_key(self):
        return '_-_'.join(
            [self.project_name, self.classifier.__class__.__name__, str(self.classifier_config), self.variation])

    def _create_path(self):
        path_name = '_-_'.join(
            [self.project_name,
             self.classifier.__class__.__name__,
             self.variation,
             str(hash(str(self.classifier_config)))]) + ".joblib"
        return str(Path(self.base_path, path_name))


class Classifier:
    def __init__(self, project_name: str, classifier: ClassVar, classifier_config: Dict, variation="common"):
        self.project_name = project_name
        self.classifier = classifier()
        self.classifier_config = classifier_config
        self.variation = variation

    def __repr__(self):
        return f'Classifier({self.project_name}, {self.classifier.__class__.__name__}, {str(self.classifier_config)})'

    def fit(self, X, y):
        with ModelCache(self.project_name, self.classifier, self.classifier_config, self.variation) as mc:
            self.classifier = mc.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)


class OverSample:
    def __init__(self, type="smote", args=None):
        assert type in ["smote"]
        self.args = dict() if args is None else args
        if type == "smote":
            self._run = self.smote

    def __call__(self, X, y):
        return self._run(X, y, self.args)

    @staticmethod
    def smote(X, y, args):
        return SMOTE(**args).fit_resample(X, y)


class CrossProjectApproach:
    pass


class All(CrossProjectApproach):
    def __call__(self, model: 'CrossProjectModel'):
        approaches = list(filter(lambda subs: subs != self.__class__, self.__class__.__base__.__subclasses__()))
        results = [approach()(model) for approach in approaches]
        return [sublist for item in results for sublist in item]


class Normalization(CrossProjectApproach):
    """
    Source
    ===
    S. Watanabe, H. Kaiya, and K. Kaijiri, ‘Adapting a fault prediction model to allow inter language reuse’,
    in Proceedings of the 4th international workshop on Predictor models in software engineering  -
    PROMISE ’08, Leipzig, Germany, 2008, p. 19, doi: 10.1145/1370788.1370794.
    """

    def __init__(self, compensate="train"):
        assert compensate in ["train", "target"]
        self.compensate = compensate

    def __call__(self, model: 'CrossProjectModel'):
        classifiers = [[(classifier, conf) for conf in model.classifiers[classifier]] for classifier in
                       model.classifiers]
        classifiers = [sublist for item in classifiers for sublist in item]
        configurations = product(model.train_projects, classifiers, model.evaluators)

        return [(
            model.target_project.name,  # target_project name
            str(self.__class__.__name__),  # approach name
            configuration[0].name,  # train_project name
            configuration[1][0].__name__,  # classifier
            configuration[1][1],  # classifier configuration
            configuration[2].__name__,  # evaluator
            self.normalization(configuration[0], model.target_project,
                               configuration[1][0], configuration[1][1],
                               configuration[2]))  # evaluation result
            for configuration in configurations]

    def normalization(self, train_project: Project, target_project, classifier, classifier_conf, evaluator):
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")
        X_train = self.encode(X_train)
        X_test = self.encode(X_test)

        def compensate_column(A: List[int], B: List[int]):
            AVG_A = np.mean(A)
            AVG_A = 10e-10 if AVG_A == 0 else AVG_A
            AVG_B = np.mean(B)
            return [(a * AVG_B) / AVG_A for a in A]

        if self.compensate == 'target':
            variation = "common"
            X_test = np.array(
                [compensate_column(A=X_test.T[column], B=X_train.T[column]) for column in range(X_test.shape[1])]).T
        else:
            variation = "normalization"
            X_train = np.array(
                [compensate_column(A=X_train.T[column], B=X_test.T[column]) for column in range(X_train.shape[1])]).T

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        oversample = OverSample()
        X_train, y_train = oversample(X_train, y_train)
        X_test, y_test = oversample(X_test, y_test)
        y_pred = Classifier(train_project.name, classifier, classifier_conf, variation).fit(X_train, y_train).predict(
            X_test)
        return evaluator(y_test, y_pred)

    @staticmethod
    def encode(data_set: List[List]):
        return PCA().fit_transform(data_set)


class KNN(CrossProjectApproach):
    """
    source
    ===
    B. Turhan, T. Menzies, A. B. Bener, and J. Di Stefano,
    ‘On the relative value of cross-company and within-company data for defect prediction’,
    Empirical Software Eng, vol. 14, no. 5, pp. 540–578, Oct. 2009, doi: 10.1007/s10664-008-9103-7.
    """

    class TrainingDataset:
        def __init__(self, train_projects: List[Project], distance: Callable, k: int = 10):
            self.X = [item for sublist in [train_project.get_X() for train_project in train_projects] for item in
                      sublist]
            self.y = [item for sublist in [train_project.get_y() for train_project in train_projects] for item in
                      sublist]
            self.distance: Callable = distance
            self.k: int = k
            self._selected_X = np.array([])
            self._selected_y = np.array([])
            self._distance_cache = dict()

        def __len__(self):
            return np.size(self.X)

        def __getitem__(self, key: int):
            assert isinstance(key, int)
            return self.X[key]

        def __repr__(self):
            return f'TrainingDataset(train_X={self.X}, train_y={self.y}, distance={self.distance}) -> selected={self._selected_X}'

        def pop(self, index: int = -1):
            return self.X.pop(index), self.y.pop(index)

        def append(self, X, y):
            self._selected_X = np.append(self._selected_X, X, axis=0)
            self._selected_y = np.append(self._selected_y, y, axis=0)

        @property
        def selected(self):
            return self._selected_X, self._selected_y

        def select_top_k(self, B):
            distances = np.array([self.calculate_distance(A, B) for A in self.X])
            indices = sorted(np.argpartition(distances, -self.k)[-self.k:],
                             reverse=True)  # get indices for top and sort in reverse order
            [self.append(*self.pop(index)) for index in indices]

        def calculate_distance(self, A, B):
            return self._lookup(self.distance)(A, B)

        def _lookup(self, func):
            def _lookup_func(A, B):
                if str(A) in self._distance_cache.keys():
                    distance = self._distance_cache[str(A)]
                else:
                    distance = func(A, B)
                    self._distance_cache[str(A)] = distance
                return distance
            return _lookup_func

    def __init__(self, distance="cosine", k=10):
        distances = {"cosine": cosine, "jaccard": jaccard, "tanimoto": rogerstanimoto}
        assert distance in distances.keys()
        self.distance = distances[distance]
        self.k = k
        self.all = None

    def __call__(self, model: 'CrossProjectModel'):
        self.train_dataset = self.TrainingDataset(model.train_projects, self.distance, k=self.k)
        X_test, y_test = model.target_project.get_set()
        [self.train_dataset.select_top_k(test_row) for test_row in X_test]
        X_train, y_train = self.train_dataset.selected
        self.classifiers = [sublist for item in self.classifiers for sublist in item]
        configurations = product(self.classifiers, model.evaluators)
        return [(
            model.target_project.name,  # target_project name
            str(self.__class__.__name__),  # approach name
            f'k={self.k} top instances for each row',  # train_project name
            configuration[0][0].__name__,  # classifier
            configuration[0][1],  # classifier configuration
            configuration[1].__name__,  # evaluator
            self.evaluate(X_train, y_train, X_test, y_test,  # dataset
                          configuration[0][0], configuration[0][1],  # classifier
                          configuration[1]))  # evaluator
            for configuration in configurations]

    @staticmethod
    def evaluate(target_project_name,
                 X_train, y_train, X_test, y_test,
                 classifier, classifier_config,
                 evaluator):
        oversample = OverSample()
        X_train, y_train = oversample(X_train, y_train)
        X_test, y_test = oversample(X_test, y_test)
        y_pred = Classifier(f'KNN: {target_project_name}', classifier, classifier_config).fit(X_train, y_train).predict(
            X_test)
        return evaluator(y_test, y_pred)


class Clustering(CrossProjectApproach):
    """
    source
    ===
    T. Menzies, A. Butcher, A. Marcus, T. Zimmermann, and D. Cok, ‘Local vs. global models for effort estimation
    and defect prediction’, in 2011 26th IEEE/ACM International Conference on Automated Software Engineering (ASE 2011),
     Lawrence, KS, USA, Nov. 2011, pp. 343–351, doi: 10.1109/ASE.2011.6100072.
    """

    def __call__(self, model: 'CrossProjectModel'):
        return [("a", "v", "g"), ("a", "g", "r")]
        pass


class BestOfBreed(CrossProjectApproach):
    """
    Source
    ===
    J. Guo, M. Rahimi, J. Cleland-Huang, A. Rasin, J. H. Hayes, and M. Vierhauser, ‘Cold-start software analytics’,
    in Proceedings of the 13th International Workshop on Mining Software Repositories -
    MSR ’16, Austin, Texas, 2016, pp. 142–153, doi: 10.1145/2901739.2901740.
    """

    def __init__(self, breed_evaluator: Callable = partial(fbeta_score, beta=2)):
        self.breed_evaluator = breed_evaluator

    def __call__(self, model: 'CrossProjectModel'):
        self.training_projects = model.train_projects
        self.target_project = model.target_project
        self.classifiers = [[(classifier, conf) for conf in model.classifiers[classifier]] for classifier in
                            model.classifiers]
        self.classifiers = [sublist for item in self.classifiers for sublist in item]
        self.evaluators = model.evaluators
        self.scores = dict([self.evaluate_breed(train_project) for train_project in self.training_projects])
        self.best_breed = max(self.scores.items(), key=itemgetter(1))[0]
        self.configurations = product(self.classifiers, self.evaluators)
        return [
            (self.target_project.name,
             str(self.__class__.__name__),
             self.best_breed.name,
             configuration[0][0].__name__,
             configuration[0][1],
             configuration[1].__name__,
             self.evaluate(self.best_breed,
                           Classifier(self.best_breed.name, configuration[0][0], configuration[0][1]),
                           configuration[1]))
            for configuration in self.configurations]

    def evaluate_breed(self, train_project: Project):
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = self.target_project.get_set(strategy="all")
        classifiers = [Classifier(train_project.name, classifier[0], classifier[1]) for classifier in self.classifiers]
        [classifier.fit(X_train, y_train) for classifier in classifiers]
        y_preds = [classifier.predict(X_test) for classifier in classifiers]
        breed_score = np.mean([self.breed_evaluator(y_test, y_pred) for y_pred in y_preds])
        return train_project, breed_score

    def evaluate(self, train_project: Project, classifier: Classifier, evaluator: Callable):
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = self.target_project.get_set(strategy="all")
        y_pred = classifier.fit(X_train, y_train).predict(X_test)
        return evaluator(y_test, y_pred)


class ProfileDriven(CrossProjectApproach):
    """
    Source
    ===
    J. Guo, M. Rahimi, J. Cleland-Huang, A. Rasin, J. H. Hayes, and M. Vierhauser, ‘Cold-start software analytics’,
    in Proceedings of the 13th International Workshop on Mining Software Repositories -
    MSR ’16, Austin, Texas, 2016, pp. 142–153, doi: 10.1145/2901739.2901740.
    """

    def __call__(self, model: 'CrossProjectModel'):
        return [("a", "v", "g"), ("a", "g", "r")]
        pass


class ELM(CrossProjectApproach):
    def __call__(self, model: 'CrossProjectModel'):
        return [("a", "v", "g"), ("a", "g", "r")]
        pass


class CrossProjectModel:
    def __init__(self, train_projects: List[Project], target_project: Project,
                 classifiers: Dict, evaluators: List[Callable]):
        self.train_projects = train_projects
        self.target_project = target_project
        self.classifiers = classifiers
        self.evaluators = evaluators

    def __repr__(self):
        return f'CrossProjectModel(train_projects[{len(self.train_projects)}], {self.target_project}, ' \
               f'{self.classifiers}, {self.evaluators})'

    def __call__(self, approaches: Union[List[CrossProjectApproach], CrossProjectApproach] = All()):
        if isinstance(approaches, All):
            return approaches(self)
        else:
            return [sublist for item in [approach(self) for approach in approaches] for sublist in item]


class Classifiers(Enum):
    RandomForest = auto(), RandomForestClassifier, [{}]
    SupportVectorMachine = auto(), SVC, [{}]
    MultilayerPerceptron = auto(), MLPClassifier, [{}]
    DecisionTree = auto(), DecisionTreeClassifier, [{}]
    NaiveBayes = auto(), GaussianNB, [{}]

    @property
    def classifier(self) -> ClassVar:
        return self.value[1]

    @property
    def configurations(self) -> List[Dict]:
        return self.value[2]


class Evaluators(Enum):
    AUC = auto(), roc_auc_score
    F1_Score = auto(), f1_score
    Precision = auto(), precision_score
    Recall = auto(), recall_score
    BrierScore = auto(), brier_score_loss

    @property
    def evaluator(self) -> Callable:
        return self.value[1]


class StoreResults:
    def __init__(self, out_path=Path(Path(__file__).parent.parent, "out")):
        self.out_path = out_path
        Path(out_path).mkdir(exist_ok=True, parents=True)
        self.path = str(Path(out_path, "results.csv"))
        self.column_names = ['Target Project', 'Approach', 'Train Project',
                             'Classifier', 'Classifier Configuration', 'Evaluator', 'Value']
        if os.path.exists(self.path):
            with open(self.path, 'w') as results_file:
                results_file.write(",".join(self.column_names) + os.linesep)

    def __call__(self, rows):
        with open(self.path, 'a') as results_file:
            [results_file.write(",".join(row) + os.linesep) for row in rows]


class Configuration:
    def __init__(self):
        self.data_path = Path(Path(__file__).parent.parent, "data", "datasets.csv")
        self.datasets = dict()
        self.get_datasets()
        self.store_results = StoreResults()

    def get_datasets(self):
        with open(self.data_path, 'r') as read_obj:
            csv_reader = DictReader(read_obj)
            for row in csv_reader:
                name = row['Project']
                version = row['Version']
                file = (row['File'], row['Class'])
                not_features = ['Project', 'Version', 'File', 'Class', 'Bugged']
                features = {column: row[column] for column in row if column not in not_features}
                bugged = row['Bugged']
                self.datasets.setdefault(name, Project(name)).append(version, file, features, bugged)

    def __iter__(self):
        self.projects = list(sorted(self.datasets.keys()))
        self.target_projects = self.projects[:]
        self.evaluators = list(map(lambda x: x.evaluator, Evaluators))
        self.classifiers = dict(zip(
            list(map(lambda x: x.classifier, Classifiers)),
            list(map(lambda x: x.configurations, Classifiers))))
        return self

    def __next__(self):
        target_project_name = self.target_projects.pop(0)
        projects = list(filter(lambda project: project != target_project_name, self.projects))
        train_projects = list(map(lambda name: self.datasets[name], projects))
        target_project = self.datasets[target_project_name]
        rows = CrossProjectModel(train_projects, target_project, self.classifiers, self.evaluators)([KNN()])
        self.store_results(rows)


if __name__ == '__main__':
    c = iter(Configuration())
    next(c)()
