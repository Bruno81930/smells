-- View that shows the average and p-value of each approach for both the f1-score and precision-recall score.
-- DecisionTreeClassifier.

create view u_s_01 as

with tmp_u_s_01_1 as
    (select Approach, Evaluator, Dataset, avg(Value) as Value
from cross_project_results
where (Evaluator="f1_score" or Evaluator="precision_recall_curve") and (Dataset="Metrics" or Dataset="Smells") and (Classifier = "DecisionTreeClassifier" or Classifier = "GenELMClassifier") and (Approach != "TestSetNormalization" and Approach != "TrainSetNormalization")
group by Approach, Classifier, Dataset, Evaluator
order by Approach, Evaluator, Dataset),

tmp_u_s_01_2 as
    (select approach as Approach, evaluator as Evaluator, p_value
from cross_project_significance
where (evaluator="f1_score" or evaluator="precision_recall_curve") and (y="Metrics" and x="Smells") and (classifier = "DecisionTreeClassifier" or classifier = "GenELMClassifier") and (approach != "TestSetNormalization" and approach != "TrainSetNormalization")),

tmp_u_s_01_3 as
    (select Approach, Evaluator, (t.Smells - t.Metrics)*100 as Value
from (
         select Approach, Evaluator,
                Sum(IF((Dataset = 'Smells'), Value, 0))  as Smells,
                Sum(IF((Dataset = 'Metrics'), Value, 0)) as Metrics
         from tmp_u_s_01_1
         group by Approach, Evaluator
     ) as t)

select v.Approach, v.Evaluator, v.Value, s.p_value as P_Value
from tmp_u_s_01_3 as v
inner join tmp_u_s_01_2 s on v.Approach = s.Approach and v.Evaluator = s.Evaluator;