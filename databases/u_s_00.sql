create view u_s_00 as
select Approach, Evaluator, Dataset, Value
from cross_project_results
where (Evaluator="f1_score" or Evaluator="precision_recall_curve") and (Dataset="Metrics" or Dataset="Smells") and (Classifier = "DecisionTreeClassifier" or Classifier = "GenELMClassifier") and (Approach != "TestSetNormalization" and Approach != "TrainSetNormalization")
order by Approach, Evaluator, Dataset;
