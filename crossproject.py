import logging
import os
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from csv import DictReader
from enum import Enum, auto
from functools import partial, lru_cache
from itertools import product
from numbers import Number
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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


class Row:
    def __init__(self,
                 target_project,
                 approach,
                 train_project,
                 classifier,
                 classifier_configuration,
                 evaluator,
                 score):
        self.target_project = target_project
        self.approach = approach
        self.train_project = train_project
        self.classifier = classifier
        self.classifier_configuration = classifier_configuration
        self.evaluator = evaluator
        self.score = score
        self.elements = [target_project, approach, train_project, classifier,
                         classifier_configuration, evaluator, score]

    def __iter__(self):
        for element in self.elements:
            yield element


class Classifier:
    def __init__(self, classifier, classifier_config: Dict):
        self.classifier = classifier
        self.classifier_config = classifier_config
        self.model = self.classifier()
        self.model.set_params(**self.classifier_config)

    def __str__(self):
        return f'{self.classifier.__name__}_{str(self.classifier_config).replace(" ", "_")}'

    def __repr__(self):
        return f'Classifier({self.classifier.__name__}, {str(self.classifier_config)})'

    def __call__(self):
        return self.model

    @property
    def name(self):
        return self.classifier.__name__

    @property
    def configuration(self):
        return str(self.classifier_config)


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


class Context:
    def __init__(self, train_project: str, target_project: str, approach: str):
        self.train_project = train_project
        self.target_project = target_project
        self.approach = approach

    def __repr__(self):
        return f'Context({self.train_project}, {self.target_project}, {self.approach})'

    def __str__(self):
        return f'{self.approach}_{self.train_project}'


class Dataset:
    def __init__(self, X_train, y_train, X_test, y_test, context: Context):
        self._training = {'X': X_train, 'y': y_train}
        self._testing = {'X': X_test, 'y': y_test}
        self.context = context

    @property
    def get(self):
        return self._training['X'], self._training['y'], self._testing['X'], self._testing['y']

    @property
    def training(self):
        return self._training["X"], self._training['y']

    @property
    def testing(self):
        return self._testing["X"], self._testing['y']

    def __repr__(self):
        return f'Dataset(X_train{list(self._training["X"].shape)}, y_train{list(self._training["y"].shape)}, X_test{list(self._testing["X"].shape)}, y_test{list(self._testing["y"].shape)}, {repr(self.context)})'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MemoryCache(metaclass=Singleton):
    def __init__(self):
        if not hasattr(self, 'cache'):
            self.cache = dict()

    def __getitem__(self, path) -> Classifier:
        if path in self.cache.keys():
            return self.cache[path]
        raise KeyError("There is no path in store with that classifier.")

    def __setitem__(self, path: str, classifier: Classifier):
        assert isinstance(classifier, Classifier), "Classifier not type classifier"
        self.cache[path] = classifier


class ModelCache:
    def __init__(self, classifier: Classifier, context: Context):
        self._classifier = classifier
        self.context = context
        self._path: Path = self.path()
        self._cached = {'disk': False, 'memory': False}

    @property
    def classifier(self):
        return self._classifier()

    def __call__(self):
        return self._cached['disk']

    def path(self):
        dir_path = Path(Path(__file__).parent, "data", "cache", str(self.context))
        dir_path.mkdir(exist_ok=True, parents=True)
        path = Path(dir_path, f'{str(self._classifier)}.joblib')
        return path

    def load(self, path):
        try:
            classifier = MemoryCache()[path]
            self._cached['memory'] = True
            return classifier
        except KeyError:
            return load(path)

    def dump(self, classifier, path):
        if not self._cached['disk']:
            dump(classifier, path)
            MemoryCache()[path] = classifier

        if self._cached['disk'] and not self._cached['memory']:
            MemoryCache()[path] = classifier

    def __enter__(self):
        if self._path.exists():
            self._cached['disk'] = True
            self._classifier = self.load(self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self.dump(self._classifier, self._path)
            return False
        return True


class Model:
    def __init__(self, classifier: Classifier, evaluator: Callable, context: Context):
        self.classifier = classifier
        self.evaluator = evaluator
        self.context = context
        self.oversample = OverSample()

    def __call__(self, dataset: Dataset) -> Number:
        X_train, y_train = self.oversample(*dataset.training)
        X_test, y_test = self.oversample(*dataset.testing)
        with ModelCache(self.classifier, self.context) as cache:
            if not cache():
                cache.classifier.fit(X_train, y_train)
            y_pred = cache.classifier.predict(X_test)
            return self.evaluator(y_test, y_pred)

    def __repr__(self):
        return f'Model({repr(self.classifier)}, {self.evaluator.__name__}, {repr(self.context)})'


class Configurations:
    def __init__(self, classifiers: List[Classifier], evaluators: List[Callable], train_projects=None):
        self._classifiers = classifiers
        self._evaluators = evaluators
        self._train_projects = train_projects

        configurations = (classifiers,
                          evaluators,
                          train_projects) if train_projects is not None else (classifiers, evaluators)
        self._configurations = product(*configurations)

    def __iter__(self):
        for configuration in self._configurations:
            element = namedtuple("Configuration", ["classifier", "evaluator", "train_project"])
            element.classifier = configuration[0]
            element.evaluator = configuration[1]
            element.train_project = configuration[2] if self._train_projects is not None else None
            yield element


class CrossProjectApproach:
    pass


class All(CrossProjectApproach):

    def subclasses(self, cls):
        ans = set()
        if cls.__subclasses__():
            for c in cls.__subclasses__():
                ans = ans.union(self.subclasses(c))
        else:
            ans = {cls}
        return ans

    def __call__(self, model: 'CrossProjectModel'):
        approaches = self.subclasses(CrossProjectApproach)
        approaches = list(filter(lambda approach: approach != All, approaches))
        results = [approach()(model) for approach in approaches]
        return [sublist for item in results for sublist in item]


class Standard(CrossProjectApproach):
    def __call__(self, model: 'CrossProjectModel'):
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        configurations = Configurations(classifiers, model.evaluators, model.train_projects)
        return [Row(
            target_project=model.target_project.name,
            approach=str(self.__class__.__name__),
            train_project=configuration.train_project.name,
            classifier=configuration.classifier.name,
            classifier_configuration=configuration.classifier.configuration,
            evaluator=configuration.evaluator.__name__,
            score=self.evaluate(configuration.classifier, configuration.evaluator,
                                model.target_project, configuration.train_project))
            for configuration in configurations]

    def evaluate(self, classifier: Classifier, evaluator, target_project, train_project):
        context = Context(train_project=train_project.name, target_project=target_project.name,
                          approach=self.__class__.__name__)

        model = Model(classifier, evaluator, context)
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")
        dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, context=context)

        return model(dataset)


class Normalization(CrossProjectApproach, ABC):
    """
    Source
    ===
    S. Watanabe, H. Kaiya, and K. Kaijiri, ‘Adapting a fault prediction model to allow inter language reuse’,
    in Proceedings of the 4th international workshop on Predictor models in software engineering  -
    PROMISE ’08, Leipzig, Germany, 2008, p. 19, doi: 10.1145/1370788.1370794.
    """

    def __call__(self, model: 'CrossProjectModel'):
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        configurations = Configurations(classifiers, model.evaluators, model.train_projects)

        return [Row(
            target_project=model.target_project.name,
            approach=str(self.__class__.__name__),
            train_project=configuration.train_project.name,
            classifier=configuration.classifier.name,
            classifier_configuration=configuration.classifier.configuration,
            evaluator=configuration.evaluator.__name__,
            score=self.normalization(configuration.train_project, model.target_project,
                                     configuration.classifier,
                                     configuration.evaluator))
            for configuration in configurations]

    def normalization(self, train_project: Project, target_project: Project, classifier: Classifier,
                      evaluator: Callable):
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")

        X_train = self.encode(X_train)
        X_test = self.encode(X_test)

        X_train, X_test = self.compensate(X_train, X_test)

        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

        context = Context(train_project.name, target_project.name, self.__class__.__name__)

        dataset = Dataset(X_train, y_train, X_test, y_test, context)
        model = Model(classifier, evaluator, context)
        return model(dataset)

    @abstractmethod
    def compensate(self, X_train, Y_test):
        pass

    @staticmethod
    def encode(data_set: List[List]):
        return PCA().fit_transform(data_set)

    @staticmethod
    def compensate_column(A: List[int], B: List[int]):
        AVG_A = np.mean(A)
        AVG_A = 10e-10 if AVG_A == 0 else AVG_A
        AVG_B = np.mean(B)
        return [(a * AVG_B) / AVG_A for a in A]


class TestSetNormalization(Normalization):
    def compensate(self, X_train, X_test):
        X_test = np.array(
            [self.compensate_column(A=X_test.T[column], B=X_train.T[column])
             for column in range(X_test.shape[1])]).T
        return X_train, X_test


class TrainSetNormalization(Normalization):
    def compensate(self, X_train, X_test):
        X_train = np.array(
            [self.compensate_column(A=X_train.T[column], B=X_test.T[column])
             for column in range(X_train.shape[1])]).T
        return X_train, X_test


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
            self._selected_X = list()
            self._selected_y = list()

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
            self._selected_X.append(X)
            self._selected_y.append(y)

        @property
        def selected(self):
            return np.array(self._selected_X), np.array(self._selected_y)

        def select_top_k(self, B):
            distances = np.array([self.calculate_distance(A.tobytes(), B.tobytes()) for A in self.X])
            indices = sorted(np.argpartition(distances, -self.k)[-self.k:],
                             reverse=True)  # get indices for top and sort in reverse order
            [self.append(*self.pop(index)) for index in indices]

        @lru_cache
        def calculate_distance(self, A, B):
            return self.distance(np.frombuffer(A, dtype=int), np.frombuffer(B, dtype=int))

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
        return [Row(
            target_project=model.target_project.name,
            approach=str(self.__class__.__name__),
            train_project=f'k={self.k} top instances for each row',
            classifier=configuration[0][0].__name__,
            classifier_configuration=configuration[0][1],
            evaluator=configuration[1].__name__,
            score=self.evaluate(model.target_project.name,
                                X_train, y_train, X_test, y_test,
                                configuration[0][0], configuration[0][1],
                                configuration[1], ))
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
        return [Row(
            target_project=self.target_project.name,
            approach=str(self.__class__.__name__),
            train_project=self.best_breed.name,
            classifier=configuration[0][0].__name__,
            classifier_configuration=configuration[0][1],
            evaluator=configuration[1].__name__,
            score=self.evaluate(self.best_breed,
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
                 classifiers: List[Tuple], evaluators: List[Callable]):
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
    NaiveBayes = auto(), GaussianNB, [{}, {}, {}]

    @property
    def classifier(self) -> ClassVar:
        return self.value[1]

    @property
    def configurations(self) -> List[Dict]:
        return self.value[2]

    @property
    def models(self) -> List[Tuple]:
        return [(self.value[1], configuration) for configuration in list(self.value[2])]


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
    def __init__(self, out_path=Path(Path(__file__).parent, "out")):
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


def timer(func):
    def timer_func(*args, **kwargs):
        logger = logging.getLogger(args[0].__class__.__name__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug(f'{func.__name__} took {time.perf_counter() - t0:0.4f} seconds')
        return result

    return timer_func


class Configuration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_path = Path(Path(__file__).parent, "data", "datasets.csv")
        self.datasets = dict()
        self.get_datasets()
        self.store_results = StoreResults()

    @timer
    def get_datasets(self):
        self.logger.debug("Obtaining datasets.")
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
        self.classifiers = [item for sublist in [classifier.models for classifier in Classifiers] for item in sublist]
        self.logger.debug(f'Beginning Configuration Iterator.')
        return self

    def __next__(self):
        target_project_name = self.target_projects.pop(0)
        projects = list(filter(lambda project: project != target_project_name, self.projects))
        train_projects = list(map(lambda name: self.datasets[name], projects))
        target_project = self.datasets[target_project_name]
        rows = CrossProjectModel(train_projects, target_project, self.classifiers, self.evaluators)()
        self.store_results(rows)


if __name__ == '__main__':
    c = iter(Configuration())
    next(c)()
    pass
