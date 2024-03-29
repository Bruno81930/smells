import logging
import os
import sys
import time
from traceback import print_tb
from abc import ABC, abstractmethod
from collections import namedtuple
from csv import DictReader
from datetime import datetime
from enum import Enum, auto
from functools import partial, lru_cache, reduce
from itertools import product, chain, tee
from multiprocessing import set_start_method, Pool, get_context

from pathlib import Path
from typing import Tuple, Dict, ClassVar, Union, List, Callable

import click
import discord
import numpy as np
from alive_progress import alive_bar
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from joblib import load, dump
from kneed import KneeLocator
from pythonjsonlogger import jsonlogger
from scipy.spatial.distance import cosine, jaccard, rogerstanimoto, euclidean
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, brier_score_loss, fbeta_score, \
    precision_recall_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn_extensions.extreme_learning_machines import RBFRandomLayer, GenELMClassifier
from toolz import apply


class DatasetCleaner:
    def __init__(self, X, y):
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "sets not numpy arrays"
        self._X = self.scale_data(self.clean_infs(X))
        self._y = y

    @staticmethod
    def _has_inf(column) -> bool:
        return np.isinf(sum(column))

    @staticmethod
    def _get_max_value(column):
        finite_column = np.copy(column)
        finite_column[finite_column == np.inf] = 0
        return column, np.max(finite_column)

    @staticmethod
    def _clear_infs(column, max_value):
        column[column == np.inf] = max_value
        return column

    def clean_infs(self, dataset: np.array):
        cleared_dataset = []
        for column in dataset.T:
            if self._has_inf(column):
                column = self._clear_infs(*self._get_max_value(column))
            cleared_dataset.append(column)
        return np.array(cleared_dataset).T

    @staticmethod
    def scale_data(dataset: np.array):
        return MinMaxScaler(feature_range=(0, 1000), copy=False).fit_transform(dataset)

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y


class Project:
    def __init__(self, name, bool_features, versions=None, data=None):
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
        try:
            return np.array([[v == "True" for v in sub] for sub in x], dtype=int) if self.bool_features else np.array(x,
                                                                                                                      dtype=float)
        except:
            x = np.array(x)
            x = np.where(x == "True", 1, x)
            x = np.where(x == "False", 0, x)
            return x.astype(float)

    def _get_y(self, versions):
        """ Accessing the data from the project and extract the bugged information."""
        y = list(map(lambda _: _['bugged'],
                     filter(lambda _: _['version'] in versions, self.data)))
        return np.array([v == "True" for v in y], dtype=int)

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

        cleaned_set = DatasetCleaner(self._get_x(versions), self._get_y(versions))
        return cleaned_set.X, cleaned_set.y

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
                 scores
                 ):
        self.target_project = target_project
        self.approach = approach
        self.train_project = train_project
        self.classifier = classifier
        self.classifier_configuration = classifier_configuration
        self.precision_recall_curve = scores[0]
        self.elements = [
            [target_project,
             approach,
             train_project,
             classifier,
             classifier_configuration,
             item.evaluator,
             self.calculate(item.score) if idx == 0 else item.score] for idx, item in enumerate(scores)]
        # classifier_configuration, item.evaluator, item.score] for item in scores]

    @staticmethod
    def calculate(score):
        precision, recall, _ = score
        return auc(recall, precision)

    # def __iter__(self):
    #     for element in self.elements:
    #         yield element

    def __str__(self):
        return os.linesep.join([','.join(map(str, elements)) for elements in self.elements])


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
    def __init__(self, otype="smote", args=None):
        assert otype in ["smote"]
        self.args = dict() if args is None else args
        if otype == "smote":
            self._run = self.smote

    def __call__(self, X, y):
        try:
            return self._run(X, y, self.args)
        except ValueError:
            return X, y

    @staticmethod
    def smote(X, y, args):
        return SMOTE(**args).fit_resample(X, y)


class Context:
    def __init__(self, dataset: str, train_project: str, target_project: str, approach: str):
        self.dataset = dataset
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
        self.store()

    @property
    def get(self):
        return self._training['X'], self._training['y'], self._testing['X'], self._testing['y']

    @property
    def training(self):
        return self._training["X"], self._training['y']

    @property
    def testing(self):
        return self._testing["X"], self._testing['y']

    def store(self):
        dir_path = Path(Path(__file__).parent, "data", "dataset", self.context.dataset,
                        f'{self.context.approach}_{self.context.train_project}_{self.context.target_project}')
        try:
            dir_path.mkdir(exist_ok=False, parents=True)
            np.savetxt(Path(dir_path, "X_train.csv"), self._training['X'], delimiter=",", fmt="%s")
            np.savetxt(Path(dir_path, "y_train.csv"), self._training['y'], delimiter=',', fmt="%s")
            np.savetxt(Path(dir_path, "X_test.csv"), self._testing['X'], delimiter=",", fmt="%s")
            np.savetxt(Path(dir_path, "y_test.csv"), self._testing['y'], delimiter=',', fmt="%s")
        except FileExistsError:
            pass

    def __repr__(self):
        return f'Dataset(X_train{list(self._training["X"].shape)}, y_train{list(self._training["y"].shape)}, X_test{list(self._testing["X"].shape)}, y_test{list(self._testing["y"].shape)}, {repr(self.context)})'


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MemoryCacheFiller:
    def __init__(self, configurator):
        self.classifiers = [str(Classifier(*model)) for model in configurator.classifiers]
        self.dataset = configurator.output
        self.approaches = [approach.__class__.__name__ for approach in configurator.approaches]
        self.projects = configurator.projects
        self.configurations = product([self.dataset], self.approaches, self.projects, self.classifiers)
        self.size = len(self.approaches) * len(self.projects) * len(self.classifiers)
        self.cache = MemoryCache()

    def __call__(self):
        with alive_bar(self.size) as bar:
            for configuration in self.configurations:
                path = ModelPath(*configuration)()
                try:
                    bar()
                    self.cache[path] = load(path)
                except:
                    continue


class MemoryCache:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        if not hasattr(self, 'cache'):
            self.cache = dict()

    def __getitem__(self, path) -> Classifier:
        assert isinstance(path, str), "Path not type string"
        try:
            return self.cache[path]
        except:
            raise KeyError("There is no path in store with that classifier.")

    def __setitem__(self, path: str, classifier: Classifier):
        assert isinstance(classifier, Classifier), "Classifier not type classifier"
        assert isinstance(path, str), "Path not type string"
        self.cache[path] = classifier


class ModelPath:
    def __init__(self, dataset, approach, project, classifier):
        dir_path = Path(Path(__file__).parent, "data", "cache", dataset, f"{approach}_{project}")
        dir_path.mkdir(exist_ok=True, parents=True)
        self.path = str(Path(dir_path, f'{str(classifier)}.joblib').absolute())

    def __call__(self):
        return self.path


class ModelCache:
    def __init__(self, classifier: Classifier, context: Context, enabled: bool):
        self._classifier = classifier
        self.context = context
        self._path: str = self.path()
        self._cached = {'disk': False, 'memory': False}
        self._enabled = enabled

    @property
    def classifier(self):
        return self._classifier()

    def __call__(self):
        return self._cached['memory'] or self._cached['disk']

    def path(self):
        return ModelPath(self.context.dataset, self.context.approach, self.context.train_project, self._classifier)()

    def load(self, path):
        try:
            if not self._enabled:
                return self._fail_cache()
            classifier = MemoryCache()[path]
            self._cached['memory'] = True
            return classifier
        except KeyError:
            try:
                classifier = load(path)
                self._cached['disk'] = True
                MemoryCache()[path] = classifier
                self._cached['memory'] = True
                return classifier
            except Exception:
                return self._fail_cache()

    def _fail_cache(self):
        self._cached['memory'] = False
        self._cached['disk'] = False
        return self._classifier

    def dump(self, classifier, path):
        if not self._cached['disk']:
            dump(classifier, path)
            MemoryCache()[path] = classifier

    def __enter__(self):
        self._classifier = self.load(self._path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            self.dump(self._classifier, self._path)
            return False
        return True


class Model:
    def __init__(self, classifier: Classifier, evaluators: List[Callable], context: Context, use_cache: bool = True):
        self.classifier = classifier
        self.evaluators = evaluators
        self.context = context
        self.oversample = OverSample()
        self.use_cache = use_cache

    def __call__(self, dataset: Dataset) -> List:
        X_test, y_test = dataset.testing
        with ModelCache(self.classifier, self.context, self.use_cache) as cache:
            if not cache():
                X_train, y_train = self.oversample(*dataset.training)
                cache.classifier.fit(X_train, y_train)
            y_pred = cache.classifier.predict(X_test)
            Scores = namedtuple("Scores", ["evaluator", "score"])
            return [Scores(evaluator.__name__, evaluator(y_test, y_pred)) for evaluator in self.evaluators]

    def __repr__(self):
        return f'Model({repr(self.classifier)}, [{",".join([evaluator.__name__ for evaluator in self.evaluators])}], {repr(self.context)})'


class Configurations:
    def __init__(self, classifiers: List[Classifier], evaluators: List[Callable], train_projects=None):
        self._classifiers = classifiers
        self._evaluators = evaluators
        self._train_projects = train_projects

        self.configurations = (classifiers,
                               [evaluators],
                               train_projects) if train_projects is not None else (classifiers, [evaluators])
        self._configurations = product(*self.configurations)

    def __iter__(self):
        for configuration in self._configurations:
            element = namedtuple("Configuration", ["classifier", "evaluators", "train_project"])
            element.classifier = configuration[0]
            element.evaluators = configuration[1]
            element.train_project = configuration[2] if self._train_projects is not None else None
            yield element

    def __len__(self):
        return len(self._classifiers) * len(self._train_projects) \
            if self._train_projects is not None else len(self._classifiers)


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
        self.logger = model.logger
        self.logger.debug(
            f'Standard. {model.target_project.name.capitalize()}.')
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        self.configurations = Configurations(classifiers, model.evaluators, model.train_projects)
        rows = list()
        with alive_bar(len(self.configurations)) as bar:
            for configuration in self.configurations:
                dataset = model.dataset
                approach = str(self.__class__.__name__)
                target_project = model.target_project
                train_project = configuration.train_project
                classifier = configuration.classifier
                evaluators = configuration.evaluators
                print(train_project.name)
                scores = self.evaluate(classifier, evaluators, target_project, train_project, dataset)
                row = Row(
                    target_project=target_project.name,
                    approach=approach,
                    train_project=configuration.train_project.name,
                    classifier=classifier.name,
                    classifier_configuration=classifier.configuration,
                    scores=scores)
                bar()
                rows.append(row)
        return rows

    def evaluate(self, classifier: Classifier, evaluators, target_project, train_project, dataset):
        context = Context(dataset=dataset, train_project=train_project.name, target_project=target_project.name,
                          approach=self.__class__.__name__)

        model = Model(classifier, evaluators, context)
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")
        dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, context=context)

        scores = model(dataset)
        return scores


class Normalization(CrossProjectApproach, ABC):
    """
    Source
    ===
    S. Watanabe, H. Kaiya, and K. Kaijiri, ‘Adapting a fault prediction model to allow inter language reuse’,
    in Proceedings of the 4th international workshop on Predictor models in software engineering  -
    PROMISE ’08, Leipzig, Germany, 2008, p. 19, doi: 10.1145/1370788.1370794.

    A. E. C. Cruz and K. Ochimizu, ‘Towards logistic regression models for predicting fault-prone code across
     software projects’, in 2009 3rd International Symposium on Empirical Software Engineering and Measurement,
     Lake Buena Vista, FL, USA, Oct. 2009, pp. 460–463, doi: 10.1109/ESEM.2009.5316002.

    """

    def __init__(self, encoding=True):
        self.encoding = encoding

    def __call__(self, model: 'CrossProjectModel'):
        self.logger = model.logger
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        self.logger.debug(
            f'{self.__class__.__name__}. {model.target_project.name.capitalize()}. {classifiers[0].name}.')

        self.configurations = Configurations(classifiers, model.evaluators, model.train_projects)
        rows = list()
        with alive_bar(len(self.configurations)) as bar:
            for configuration in self.configurations:
                dataset = model.dataset
                approach = str(self.__class__.__name__)
                classifier = configuration.classifier
                evaluators = configuration.evaluators
                target_project = model.target_project
                train_project = configuration.train_project
                scores = self.normalization(train_project, target_project, classifier, evaluators, dataset)
                row = Row(
                    target_project=target_project.name,
                    approach=approach,
                    train_project=train_project.name,
                    classifier=classifier.name,
                    classifier_configuration=classifier.configuration,
                    scores=scores
                )
                bar()
                rows.append(row)
        return rows

    def normalization(self, train_project: Project, target_project: Project, classifier: Classifier,
                      evaluators: List[Callable], dataset: str):
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")

        X_train = self.encode(X_train)
        X_test = self.encode(X_test)

        X_train, X_test = self.compensate(X_train, X_test)

        if self.encoding:
            X_train = StandardScaler().fit_transform(X_train)
            X_test = StandardScaler().fit_transform(X_test)

        context = Context(dataset, train_project.name, target_project.name, self.__class__.__name__)

        dataset = Dataset(X_train, y_train, X_test, y_test, context)
        model = Model(classifier, evaluators, context)
        scores = model(dataset)
        return scores

    @abstractmethod
    def compensate(self, X_train, Y_test):
        pass

    def encode(self, data_set: List[List]):
        if self.encoding:
            return PCA().fit_transform(data_set)
        return data_set

    @staticmethod
    def compensate_column(A: List[int], B: List[int]):
        AVG_A = np.mean(A)
        AVG_A = 10e-10 if AVG_A == 0 else AVG_A
        AVG_B = np.mean(B)
        x = [(a * AVG_B) / AVG_A for a in A]
        if np.isnan(np.sum(x)):
            pass
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

    def __init__(self, distance="cosine", k=10):
        distances = {"euclidean": euclidean, "cosine": cosine, "jaccard": jaccard, "tanimoto": rogerstanimoto}
        assert distance in distances.keys()
        self.distance = distances[distance]
        self.k = k
        self.all = None

    class TrainingDatasetSelector:

        def __init__(self, dataset, logger, train_projects: List[Project], distance: Callable, k: int = 10):
            self.X = [item for sublist in [train_project.get_X() for train_project in train_projects] for item in
                      sublist]
            self.y = [item for sublist in [train_project.get_y() for train_project in train_projects] for item in
                      sublist]
            self.dataset = dataset
            self.distance: Callable = distance
            self.logger = logger
            self.k: int = k
            self.cluster = self.Cluster(self.X)
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

        class Cluster:
            # number of clusters for smells 11
            def __init__(self, X, n_clusters_range=(1, 51)):
                self.X = X
                self.n_clusters_range = n_clusters_range
                self.kmeans = self._calculate_k_means()
                self.clusters = self._map_clusters_to_vectors()

            def _calculate_k_means(self):
                kmeans_range = {n_clusters:
                                # Used Normal KMeans for Smells computation. MiniBatch was bc memory problems
                                    MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(self.X)
                                for n_clusters in range(*self.n_clusters_range)}
                wws = [kmean.inertia_ for kmean in kmeans_range.values()]
                clusters = [n_clusters for n_clusters in kmeans_range.keys()]
                opt_n_clusters = KneeLocator(clusters, wws, curve='convex', direction='decreasing').knee
                return kmeans_range[opt_n_clusters]

            def _map_clusters_to_vectors(self):
                vectors = list(map(lambda l: list(l), self.X))
                clusters = list(self.kmeans.predict(self.X))
                mapping = dict()
                [mapping.setdefault(n_clusters, []).append(vectors) for (n_clusters, vectors) in zip(clusters, vectors)]
                return mapping

            def predict(self, X):
                cluster = self.kmeans.predict([X])[0]
                return np.array(list([np.array(x) for x in self.clusters[cluster]]))

            def __repr__(self):
                return f'Cluster(X, n_clusters_range={self.n_clusters_range})'

        def select_top_k(self, B):
            X = self.cluster.predict(B)
            distances = np.array([self._calculate_distance(A.tobytes(), B.tobytes()) for A in X])
            indices = sorted(np.argpartition(distances, -self.k)[-self.k:],
                             reverse=True)  # get indices for top and sort in reverse order
            [self.append(*self.pop(index)) for index in indices]

        def __call__(self, from_testing_rows):
            self.logger.debug("KNN Selecting from testing rows")
            with alive_bar(len(from_testing_rows)) as bar:
                for test_row in from_testing_rows:
                    self.select_top_k(test_row)
                    bar()

            return np.array(self._selected_X), np.array(self._selected_y)

        @lru_cache(maxsize=256)
        def _calculate_distance(self, A, B):
            return self.distance(np.frombuffer(A, dtype=int), np.frombuffer(B, dtype=int))

    def __call__(self, model: 'CrossProjectModel'):
        model.logger.debug(
            f'KNN. {model.target_project.name.capitalize()}.')

        self.train_dataset = self.TrainingDatasetSelector(model.dataset, model.logger, model.train_projects,
                                                          self.distance, k=self.k)
        X_test, y_test = model.target_project.get_set()
        X_train, y_train = self.train_dataset(X_test)
        train_project_name = f'k={self.k} top instances for each row'
        context = Context(model.dataset, train_project_name, model.target_project.name, self.__class__.__name__)
        dataset = Dataset(X_train, y_train, X_test, y_test, context)
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        self.configurations = Configurations(classifiers, model.evaluators)

        rows = list()

        with alive_bar(len(self.configurations)) as bar:
            for configuration in self.configurations:
                approach = str(self.__class__.__name__)
                classifier = configuration.classifier
                evaluators = configuration.evaluators
                train_project = f'k={self.k} top instances for each row'
                target_project = model.target_project
                scores = Model(classifier, evaluators, context, use_cache=False)(dataset)

                row = Row(
                    target_project=target_project.name,
                    approach=approach,
                    train_project=train_project,
                    classifier=classifier.name,
                    classifier_configuration=classifier.configuration,
                    scores=scores
                )
                bar()
                rows.append(row)

        return rows


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
        self.breed_evaluator.__name__ = "f2-score"

    def __call__(self, model: 'CrossProjectModel'):
        model.logger.debug(
            f'BestOfBreed. {model.target_project.name.capitalize()}. ')

        dataset = model.dataset
        approach = str(self.__class__.__name__)
        train_projects = model.train_projects
        target_project = model.target_project
        classifiers = [Classifier(*classifier) for classifier in model.classifiers]
        self.configurations = Configurations(classifiers, model.evaluators)

        rows = list()
        with alive_bar(len(classifiers)) as bar:
            for classifier in classifiers:
                train_project = self.evaluate_breed(dataset, train_projects, classifier)
                evaluators = model.evaluators
                scores = self.evaluate(dataset, classifier, evaluators, train_project, target_project)
                row = Row(
                    target_project=model.target_project.name,
                    approach=approach,
                    train_project=train_project.name,
                    classifier=classifier.name,
                    classifier_configuration=classifier.configuration,
                    scores=scores
                )
                bar()
                rows.append(row)

        return rows

    def evaluate(self, dataset, classifier, evaluators, train_project, target_project):
        context = Context(dataset, train_project.name, target_project.name, self.__class__.__name__)
        X_train, y_train = train_project.get_set(strategy="all")
        X_test, y_test = target_project.get_set(strategy="all")
        dataset = Dataset(X_train, y_train, X_test, y_test, context)
        model = Model(classifier, evaluators, context)
        scores = model(dataset)
        return scores

    def evaluate_breed(self, dataset: str, train_projects: List[Project], classifier: Classifier):
        def calculate(train_project):
            nonlocal dataset
            nonlocal classifier
            test_projects = list(filter(lambda project: project != train_project, train_projects))
            X_train, y_train = train_project.get_set(strategy="all")
            test_sets = {test_project.name: test_project.get_set(strategy="all") for test_project in test_projects}
            datasets = [Dataset(X_train,
                                y_train,
                                *test_sets[test_project],
                                Context(dataset, train_project.name, test_project, self.__class__.__name__))
                        for test_project in test_sets.keys()]
            model = Model(classifier, [self.breed_evaluator], datasets[0].context)
            return np.mean([model(dataset)[0].score for dataset in datasets])

        scores = dict()
        with alive_bar(len(train_projects)) as bar:
            for project in train_projects:
                scores[project] = calculate(project)
                bar()

        return max(scores, key=scores.get)


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
    """
    Source
    ===
    Bal, P.R. and Kumar, S., 2018. Cross Project Software Defect Prediction using Extreme Learning Machine:
    An Ensemble based Study. In ICSOFT (pp. 354-361).
    """

    def __call__(self, model: 'CrossProjectModel'):
        model.logger.debug(f'ELM. {model.target_project.name.capitalize()}.')

        configuration = {"hidden_layer": RBFRandomLayer(n_hidden=10 * 2, rbf_width=0.1, random_state=0)}
        classifier = Classifier(GenELMClassifier, configuration)
        evaluators = model.evaluators
        target_project = model.target_project
        train_projects = model.train_projects
        dataset = model.dataset
        approach = str(self.__class__.__name__)
        model.logger.debug(f"Calculating for all training sets.")
        scores = self.evaluate(classifier, evaluators, target_project, train_projects, dataset, all=True)

        rows = list()
        rows.append(Row(
            target_project=model.target_project.name,
            approach=approach,
            train_project="all",
            classifier=classifier.name,
            classifier_configuration=classifier.configuration,
            scores=scores))

        model.logger.debug(f"Calculating for each training set.")
        self.configurations = Configurations([classifier], model.evaluators, model.train_projects)
        rows = list()
        with alive_bar(len(self.configurations)) as bar:
            for configuration in self.configurations:
                train_project = configuration.train_project
                evaluators = configuration.evaluators
                print(train_project.name)
                scores = self.evaluate(classifier, evaluators, target_project, [train_project], dataset)
                row = Row(
                    target_project=target_project.name,
                    approach=approach,
                    train_project=configuration.train_project.name,
                    classifier=classifier.name,
                    classifier_configuration=classifier.configuration,
                    scores=scores)
                bar()
                rows.append(row)

        return rows

    def evaluate(self, classifier: Classifier, evaluators, target_project, train_projects: List[Project],
                 dataset, all=False):
        train_project_name = f"all_{target_project.name}" if all else train_projects[0].name
        context = Context(dataset=dataset, train_project=train_project_name, target_project=target_project.name,
                          approach=self.__class__.__name__)

        model = Model(classifier, evaluators, context)
        train_sets = tee((train_project.get_set(strategy="all") for train_project in train_projects), 2)
        X_train = np.vstack([train_set[0] for train_set in train_sets[0]])
        y_train = np.concatenate([train_set[1] for train_set in train_sets[1]])
        X_test, y_test = target_project.get_set(strategy="all")
        dataset = Dataset(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, context=context)

        scores = model(dataset)
        return scores


class CrossProjectModel:
    def __init__(self, dataset, train_projects: List[Project], target_project: Project,
                 classifiers: List[Tuple], evaluators: List[Callable],
                 logger,
                 approaches: Union[List[CrossProjectApproach], CrossProjectApproach] = All()):
        self.dataset = dataset
        self.train_projects = train_projects
        self.target_project = target_project
        self.classifiers = classifiers
        self.evaluators = evaluators
        self.logger = logger
        self.approaches = approaches

    def __repr__(self):
        return f'CrossProjectModel(train_projects[{len(self.train_projects)}], {self.target_project}, ' \
               f'{self.classifiers}, {self.evaluators})'

    def __call__(self):
        def catch(func, *args, **kwargs):
            approach = func.__class__.__name__
            target_project = self.target_project.name
            try:
                Discord()(f"""
                Cross-Project Iteration: Started
                Dataset is {self.dataset}.
                Target Project is {target_project}.
                Approach is {approach}
                """)
                data = func(*args, **kwargs)
                Discord()(f"""
                Cross-Project Iteration: Finished
                Dataset is {self.dataset}.
                Target Project is {target_project}.
                Approach is {approach}
                """)
                StoreResults(self.dataset)(data)

            except Exception as e:
                Discord()(f"""
                Cross-Project Iteration: Exception
                Dataset is {self.dataset}.
                Target Project is {target_project}.
                Approach is {approach}
                Exception is {e}
                {print_tb(e.__traceback__)}
                """)
                self.logger.error(
                    f'Exception. Target Project={target_project}. Approach={approach}'
                )

        if isinstance(self.approaches, All):
            catch(self.approaches(self), self)
        else:
            [catch(approach, self) for approach in self.approaches]


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

    @property
    def models(self) -> List[Tuple]:
        return [(self.value[1], configuration) for configuration in list(self.value[2])]


class Evaluators(Enum):
    PRC = auto(), precision_recall_curve
    AUC = auto(), roc_auc_score
    F1_Score = auto(), f1_score
    Precision = auto(), precision_score
    Recall = auto(), recall_score
    BrierScore = auto(), brier_score_loss

    @property
    def evaluator(self) -> Callable:
        return self.value[1]


class Logger:
    def __init__(self, dataset: str, target_project: str, level=logging.DEBUG):
        self.dataset = dataset
        self.target_project = target_project
        self.dir_path = Path(Path(__file__).parent, "data", "logs", dataset)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.time_name = str(datetime.now()).replace(" ", "_")
        self.logger = logging.getLogger(dataset)
        self.logger.setLevel(level)
        self.logger.addHandler(self._json_file_handler())
        self.logger.addHandler(self._csv_file_handler())
        self.logger.addHandler(self._log_handler())

    def __call__(self, *args, **kwargs):
        return self.logger

    def _json_file_handler(self):
        json_file_handler = logging.FileHandler(Path(self.dir_path, f'{self.target_project}_{self.dataset}.json'))
        json_formatter = Logger.CustomJsonFormatter('%(timestamp)s %(processName)s %(level)s %(name)s %(message)s')
        json_file_handler.setFormatter(json_formatter)
        return json_file_handler

    def _csv_file_handler(self):
        csv_file_handler = logging.FileHandler(Path(self.dir_path, f'{self.target_project}_{self.dataset}.csv'))
        csv_formatter = logging.Formatter('%(asctime)s,%(processName)s,%(name)s,%(message)s')
        csv_file_handler.setFormatter(csv_formatter)
        return csv_file_handler

    @staticmethod
    def _log_handler():
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter('%(asctime)s | %(processName)s | %(name)s | %(message)s')
        log_handler.setFormatter(log_formatter)
        return log_handler

    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super(Logger.CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
            if not log_record.get('timestamp'):
                now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
                log_record['timestamp'] = now
            if log_record.get('level'):
                log_record['level'] = log_record['level'].upper()
            else:
                log_record['level'] = record.levelname


class StoreResults:
    def __init__(self, dataset, out_path=Path(Path(__file__).parent, "out")):
        self.out_path = out_path
        Path(out_path).mkdir(exist_ok=True, parents=True)
        # self.path = str(Path(out_path, f'tmp_{dataset}.csv'))
        # self.path = str(Path(out_path, f'{dataset}.csv'))
        self.path = str(Path(out_path, f'best_{dataset}.csv'))
        self.column_names = ['Target Project', 'Approach', 'Train Project',
                             'Classifier', 'Classifier Configuration', 'Evaluator', 'Value']
        if not os.path.exists(self.path):
            with open(self.path, 'w') as results_file:
                results_file.write(",".join(self.column_names) + os.linesep)

    def __call__(self, rows: List[Row]):
        with open(self.path, 'a') as results_file:
            [results_file.write(str(row) + os.linesep) for row in rows]
        Discord()(f"""
                        Cross-Project Iteration: Stored
                        Data length is {len(rows)}
                        Data head is {rows[0]}
                        Path is {self.path}.
                        """)


def timer(func):
    def timer_func(*args, **kwargs):
        logger = logging.getLogger(args[0].__class__.__name__)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        logger.debug(f'{func.__name__} took {time.perf_counter() - t0:0.4f} seconds')
        return result

    return timer_func


class Configuration:
    def __init__(self, configurator: "Configurator"):
        self.configurator = configurator
        self.boolean = configurator.output == "smells"
        self.logger = Logger(configurator.output, "configuration")()
        self.data_path = Path(Path(__file__).parent, "data", configurator.path)
        self.datasets = dict()
        self._get_datasets()
        self.store_results = StoreResults(self.configurator.output)

    def _get_datasets(self):
        self.logger.debug("Obtaining Datasets.")
        num_lines = sum(1 for line in open(self.data_path))
        with open(self.data_path, 'r') as read_obj:
            csv_reader = DictReader(read_obj)
            with alive_bar(num_lines - 1) as bar:
                for row in csv_reader:
                    name = row['Project']
                    version = row['Version']
                    file = (row['File'], row['Class'])
                    not_features = ['Project', 'Version', 'File', 'Class', 'Bugged']
                    features = {column: row[column] for column in row if column not in not_features}
                    bugged = row['Bugged']
                    self.datasets.setdefault(name, Project(name, bool_features=self.boolean)).append(version, file,
                                                                                                     features, bugged)
                    bar()

    def __iter__(self):
        self.projects = list(sorted(self.datasets.keys()))
        self.target_projects = self.projects[:]
        self.target_projects = list(filter(lambda project: project in self.configurator.projects, self.target_projects))
        if self.configurator.with_cache():
            self.logger.debug("Obtaining Classifiers from Cache.")
            MemoryCacheFiller(self.configurator)()
        self.evaluators = list(map(lambda x: x.evaluator, Evaluators))
        return self

    def __next__(self):
        try:
            target_project_name = self.target_projects.pop(0)
            projects = list(filter(lambda project: project != target_project_name, self.projects))
            train_projects = list(map(lambda name: self.datasets[name], projects))
            target_project = self.datasets[target_project_name]
            configuration = {"dataset": self.configurator.output, "train_projects": train_projects,
                             "target_project": target_project,
                             "classifiers": self.configurator.classifiers, "evaluators": self.evaluators,
                             "logger": self.logger,
                             "approaches": self.configurator.approaches}
            return configuration
        except IndexError:
            raise StopIteration


class Discord:
    active = False

    def __init__(self):
        if not Discord.active:
            return
        load_dotenv()
        webhook_id = os.getenv('DISCORD_WEBHOOK')
        self.webhook = discord.Webhook.from_url(webhook_id, adapter=discord.RequestsWebhookAdapter())

    def __call__(self, message):
        if not Discord.active:
            return
        self.webhook.send(message)


class ValueErrorMessage:
    dataset = "--dataset/-d argument value is incorrect."
    approach = "--approach/-a argument values are incorrect."
    classifier = "--classifier/-c argument values are incorrect."
    project = "--target/-t argument values are incorrect."


class Configurator:
    def __init__(self, dataset, approaches, classifier, projects, nocache):
        self._path = self.get_path(dataset)
        self._output = dataset
        self._approaches = self.get_approaches(approaches)
        self._classifiers = self.get_classifiers(classifier)
        self._projects = self.get_projects(projects)
        self._cache = True if not nocache else False

    @staticmethod
    def get_path(dataset):
        _ = {"smells": "datasets.csv", "metrics": "metrics_datasets.csv",
             "smells_metrics": "smellsmetrics_datasets.csv"}
        assert dataset in _.keys(), ValueErrorMessage.dataset
        return _[dataset]

    @staticmethod
    def get_approaches(approaches):
        _ = {"std": Standard(), "train": TrainSetNormalization(encoding=False),
             "test": TestSetNormalization(encoding=False), "knn": KNN(), "best": BestOfBreed(),
             "elm": ELM()}
        assert all([approach in _.keys() for approach in approaches]), ValueErrorMessage.approach
        return [_[approach] for approach in approaches]

    @staticmethod
    def get_classifiers(classifiers):
        _ = {"rf": (RandomForestClassifier, [{}]), "svc": (SVC, [{}]), "mp": (MLPClassifier, [{}]),
             "dt": (DecisionTreeClassifier, [{}]), "nb": (GaussianNB, [{}])}
        assert all([classifier in _.keys() for classifier in classifiers]), ValueErrorMessage.classifier
        classifiers = [_[classifier] for classifier in classifiers]
        classifiers = [list(zip([item[0]] * len(item[1]), list(item[1]))) for item in classifiers]
        return [item for sublist in classifiers for item in sublist]

    @staticmethod
    def get_projects(projects):
        if projects == ("all",):
            return Projects.projects
        else:
            assert all([project in Projects.projects for project in projects]), ValueErrorMessage.project
            return projects

    @property
    def path(self) -> str:
        return self._path

    @property
    def approaches(self) -> List:
        return self._approaches

    @property
    def output(self):
        return self._output

    @property
    def classifiers(self):
        return self._classifiers

    @property
    def projects(self):
        return self._projects

    def with_cache(self):
        return self._cache


class Projects:
    projects = ['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas', 'avro',
                'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne', 'clerezza', 'cocoon',
                'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections', 'commons-compress',
                'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang',
                'commons-math', 'commons-net', 'commons-validator', 'commons-vfs', 'continuum', 'crunch', 'curator',
                'cxf', 'deltaspike', 'directory-kerby', 'directory-server', 'directory-studio', 'drill', 'flink',
                'giraph', 'hadoop', 'hbase', 'helix', 'hive', 'isis', 'jackrabbit', 'jackrabbit-oak', 'jclouds',
                'jena', 'johnzon', 'juneau', 'kafka', 'karaf', 'knox', 'kylin', 'lucene-solr', 'manifoldcf', 'maven',
                'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'nifi', 'nutch', 'ofbiz', 'olingo-odata4',
                'openjpa', 'openmeetings', 'opennlp', 'openwebbeans', 'parquet-mr', 'phoenix', 'plc4x', 'pulsar',
                'qpid-jms', 'ranger', 'reef', 'roller', 'samza', 'santuario-java', 'servicecomb-java-chassis', 'shiro',
                'storm', 'struts', 'syncope', 'systemml', 'tajo', 'tapestry-5', 'tez', 'tika', 'tinkerpop', 'tomcat',
                'tomee', 'uima-ruta', 'wicket', 'xmlgraphics-fop', 'zeppelin']


class Help:
    dataset = "Evaluated Dataset: smells | metrics | smells_metrics"
    classifier = "Choose One or More Classifiers: Random Forest => rf | Support Vector Machine => svc | Multilayer " \
                 "Perceptron => mp | Decision Tree => dt | Naive Bayes => nb "
    approach = "Choose One or More Approaches: Standard => std | Train Set Normalization => train | Test Set " \
               "Normalization => test | KNN => knn | Best of Breed => best | ELM => elm "
    processes = "Choose Number of Processes: 1 .. *"
    projects = "Choose One or More Projects in --list. If All Projects => all]}"
    list = "Returns a list with the projects"
    nocache = "Deactivates the use of a cache for model retrieving"


@click.command()
@click.option('--dataset', '-d', default="smells", help=Help.dataset)
@click.option('--classifier', '-c', multiple=True, default=["rf", "svc", "mp", "dt", "nb"], help=Help.classifier)
@click.option('--approach', '-a', multiple=True, default=["std", "train", "test", "knn", "best", "elm"],
              help=Help.approach)
@click.option('--project', '-p', multiple=True, default=['all'], help=Help.projects)
@click.option('--list', '-l', is_flag=True, help=Help.list)
@click.option('--nocache', is_flag=True, help=Help.nocache)
def main(dataset, classifier, approach, project, list, nocache):
    if list:
        print(Projects.projects)
        return
    configurator = Configurator(dataset, approach, classifier, project, nocache)
    configs = Configuration(configurator)
    for config in configs:
        config["logger"] = Logger(dataset, config["target_project"].name)()
        try:
            CrossProjectModel(**config)()
        except Exception as e:
            tb = e.__traceback__
            Discord()(f"Fail. Dataset {dataset}. Project {config['target_project'].name}."f"Exception {print_tb(tb)}")


def test(dataset, classifier, approach, project, list, nocache):
    if list:
        print(Projects.projects)
        return
    configurator = Configurator(dataset, approach, classifier, project, nocache)
    configs = Configuration(configurator)
    for config in configs:
        config["logger"] = Logger(dataset, config["target_project"].name)()
        try:
            CrossProjectModel(**config)()
        except Exception as e:
            tb = e.__traceback__
            Discord()(f"Fail. Dataset {dataset}. Project {config['target_project'].name}."f"Exception {print_tb(tb)}")


if __name__ == '__main__':
    main()
    #test("metrics", ["rf"], ["best"], ["shiro"], False, False)
