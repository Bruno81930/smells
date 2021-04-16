from collections import namedtuple

from u_s_02 import run

if __name__ == '__main__':
    Vars = namedtuple('Vars', ['approaches', 'evaluators', 'datasets'])
    vs = Vars(approaches=["Standard", "KNN", "ELM", "BestOfBreed"],
              evaluators=["roc_auc_score"],
              datasets=["Smells", "Metrics"])
    run(vs.approaches, vs.evaluators, vs.datasets, "u_s_04_tmp", diff=False)
