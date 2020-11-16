# TODO Herbold 2013 Clustering
# TODO Guo et al. 2016 Best of Breed
# TODO Bal 2018 ELM
# TODO Watanabe et al. 2008 normalization 
# TODO Guo et al. 2016 Profile Driven based on smells -> deriving metrics from each
# TODO Menzies et al. 2011 local vs global 
# TODO Turhan 2009 KNN with metric derivation
from multiprocessing import Pool
from typing import Dict, List, Callable, Union
from numbers import Number
from itertools import chain, permutations, product, combinations
from functools import partial

from sklearn.preprocessing import LabelEncoder
from toolz import thread_last, keyfilter
import pandas as pd


class Model:
    @staticmethod
    def normalization(
            datasets: Dict,
            target_project: str,
            metrics: Dict,
            metrics_args: Dict,
            clfs: Dict,
            clfs_args: Dict,
            encoding: bool = False,

    ):

        def get_projects(_):
            return set([key[0] for key in _.keys()])

        def filter_target(_):
            return list(filter(lambda project: project != target_project, _))

        if target_project not in get_projects(datasets):
            raise ValueError("Project {} not a defined project".format(target_project))

        train_projects = thread_last(
            datasets,
            get_projects,
            filter_target)

        if encoding:
            pass

        def normalize(train_project):
            return Strategy.single_project_strategy(datasets,
                                                    train_project, target_project,
                                                    clfs, clfs_args,
                                                    metrics, metrics_args,
                                                    versioning_method="all")

        data = []
        with Pool() as p:
            data.append(p.map(normalize, train_projects))

        return pd.DataFrame(data)

    @staticmethod
    def knn(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass

    @staticmethod
    def localglobal(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass

    @staticmethod
    def clustering(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass

    @staticmethod
    def bestofbreed(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass

    @staticmethod
    def profiledriven(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass

    @staticmethod
    def elm(datasets: Dict, args: Dict = {}, metrics: List = ['f1-score']):
        pass


class Strategy:
    @staticmethod
    def single_project_strategy(
            datasets: Dict,
            train_project: str,
            target_project: str,
            classifiers: Dict,
            classifiers_args: Dict,
            metrics: Dict,
            metric_args: Dict,
            versioning_method: str = "all") -> List:
        """
        Strategy focused on applying cross-project defect prediction, given a Training project dataset and a Testing
        project dataset.
        [Example]
        Project A -> Training project
        Project B -> Target Project
         A  |  B
        vA1 | vB1
        vA2 | vB2
        vA3 | vB3
        vA4 | vB4
        vA5 | vB5

        It includes two possible options for the versioning_method:
        - all
            Trains the model with all versions of the training project and tests with all versions of the target
            project.
            Considering the example above, the training dataset includes all versions of the training dataset, i.e.
            {vA1, vA2, vA3, vA4, vA5}, and the testing dataset includes all versions of the target dataset, i.e.,
            {vB1, vB2, vB3, vB4, vB5}.
        - separate
            Trains the model with the training versions of the training project and tests with the testing version
            of the target project.
            Considering the example above, the training dataset includes the versions {vA1, vA2, vA3, vA4} and the
            testing dataset includes the version {vB5}, from the target project.

        Parameters
        ----------
        datasets : Dict of Pandas Dataframes
            should be given given by CrossProjectAnalysis in base.py
        train_project : str
            name of train project
        target_project : str
            name of target project
        classifier : Dict
            dictionary with name of classifier for dataset and the function from scikit learn that represents it.
            an example is "Naive Bayes": "NaiveBayes()"
        classifier_args : Dict
            parameters related to the classifier from scikit-learn
            an example is "Naive Bayes": {"C": 0.3, "limit":100}
        metrics : Dict
            dictionary with name of the metric for the dataset and the function from scikit learn.
            an example is "AUC": auc()
        metric_args : Dict
            parameters related to the evaluation function from scikit-learn for each metric
            an example is "AUC": {"parameter": "value"}
        versioning_method : str
            option to apply versioning {all, separate}

        Returns
        -------
        List
            A List of tuples representing the rows of the dataset.
            e.g.
                (target_project, training_project, classifier, classifier_configurations, metric, value)


        """

        def filter_datasets(train_p, target_p):
            def key_project(key, project): return key[0] == project

            key_train_project = partial(key_project, project=train_p)
            key_target_project = partial(key_project, project=target_p)
            return keyfilter(key_train_project, datasets), keyfilter(key_target_project, datasets)

        class Versioning:
            # Fills the dataset values from String Booleans ["True", "False"] to [0, 1]
            @staticmethod
            def str2bool(x):
                return 1 if x == 'True' else 0

            # Gets only the bug column from the datasets
            @staticmethod
            def bugged(df):
                return df[['Bugged']].values

            # Get all the column that are not bugs
            @staticmethod
            def notbugged(df): return df.loc[:, df.columns != 'Bugged'].values

            @staticmethod
            def all(training_dict: Dict, target_dict: Dict):
                training_dataset = pd.concat(training_dict.values()).applymap(Versioning.str2bool)
                target_dataset = pd.concat(target_dict.values()).applymap(Versioning.str2bool)
                return Versioning.notbugged(training_dataset), Versioning.bugged(training_dataset), \
                       Versioning.notbugged(target_dataset), Versioning.bugged(target_dataset)

            @staticmethod
            def separate(training_dict: Dict, target_dict: Dict):
                training_dataset = pd.concat(list(training_dict.values())[:-1]).applymap(Versioning.str2bool)
                target_dataset = list(target_dict.values())[-1].applymap(Versioning.str2bool)
                return Versioning.notbugged(training_dataset), Versioning.bugged(training_dataset), \
                       Versioning.notbugged(target_dataset), Versioning.bugged(target_dataset)

        allowed_versioning_methods = ['all', 'separate']
        if versioning_method not in allowed_versioning_methods:
            raise ValueError("The allowed methods are \"all\" and \"last-version\".")

        filter_datasets = partial(filter_datasets,
                                  train_p=train_project,
                                  target_p=target_project)
        if versioning_method == "all":
            X_train, y_train, X_test, y_test = Versioning.all(*filter_datasets())
        else:
            X_train, y_train, X_test, y_test = Versioning.separate(*filter_datasets())

        data = []
        for clf_key, clf_fn in classifiers.items():
            try:
                clf_fn.fit(X_train, y_train, **classifiers_args[clf_key])
                y_pred = clf_fn.predict(X_test)

                def calculate_metric(metric, metric_args):
                    return metric(y_test, y_pred, **metric_args)

                # create the rows for the dataset
                data.append(list(map(lambda metric_name:
                                     (target_project,
                                      train_project,
                                      clf_key,
                                      metric_name,
                                      calculate_metric(metrics[metric_name], metric_args[metric_name])
                                      )
                                     , metrics.keys())))
            except Exception:
                # TODO Add log module
                pass

        return data
