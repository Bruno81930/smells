# Create table in database with median of values for f1-score and precision recall.
# This applies for decision tree classifier

import os
from collections import namedtuple
from itertools import product
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mysql.connector import connect, Error
from sqlalchemy import create_engine

__all__ = ['run']


def filter_query(approaches, evaluators, datasets):
    def or_join(lst):
        return " or ".join(lst)
    approaches = or_join([f"Approach='{approach}'" for approach in approaches])
    evaluators = or_join([f"Evaluator='{evaluator}'" for evaluator in evaluators])
    datasets = or_join([f"Dataset='{dataset}'" for dataset in datasets])
    classifiers = ["DecisionTreeClassifier", "GenELMClassifier"]
    classifiers = or_join([f"Classifier='{classifier}'" for classifier in classifiers])
    return f"""
        select *
        from cross_project_results
        where ({approaches}) and ({evaluators}) and ({datasets}) and ({classifiers})
    """


def calculate_median(conn, query, approach, evaluator, dataset):
    cur = conn.cursor()
    query = f"""
        with tmp_table as ({query}) 
        select Value from tmp_table where Evaluator="{evaluator}" and Approach="{approach}" and Dataset="{dataset}";"""
    cur.execute(query)
    values = [value[0] for value in cur.fetchall()]
    return approach, evaluator, dataset, np.median(values)


def run(approaches, evaluators, datasets, table_name, diff=True):
    try:
        load_dotenv()
        conn = connect(host="132.72.64.83", user="machadob", password=os.getenv("DBPASS"), database="repositories")
        confs = product(approaches, evaluators, datasets)
        query = filter_query(approaches, evaluators, datasets)
        rows = [calculate_median(conn, query, *conf) for conf in confs]

        df = pd.DataFrame(rows, columns=["Approach", "Evaluator", "Dataset", "Median"])
        df.sort_values(by=["Approach", "Evaluator", "Dataset"], ascending=False, inplace=True)

        def difference(series):
            smells, metrics = list(series)
            return smells - metrics

        if diff:
            df = df.groupby(["Approach", "Evaluator"]).agg({"Median": difference}).reset_index()

        engine = create_engine(f"mysql+pymysql://machadob:{os.getenv('DBPASS')}@132.72.64.83/repositories")
        df.to_sql(table_name, con=engine, index=False)


    except Error as err:
        print("Connection failed")
    else:
        print("Connection successful")
    finally:
        conn.close()


if __name__ == '__main__':
    Vars = namedtuple('Vars', ['approaches', 'evaluators', 'datasets'])
    vs = Vars(approaches=["Standard", "KNN", "ELM", "BestOfBreed"],
              evaluators=["f1_score", "precision_recall_curve"],
              datasets=["Smells", "Metrics"])
    run(vs.approaches, vs.evaluators, vs.datasets, "u_s_02_tmp")
