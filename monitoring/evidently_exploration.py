import pandas as pd
import json
from sklearn import datasets

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import (
    DataDriftTab,
    CatTargetDriftTab
)

iris = datasets.load_iris()
iris_frame = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_frame['target'] = iris.target

print("DataFrame Shape:{}".format(iris_frame.shape))


#shuffle the dataframe
iris_frame_shuffled = iris_frame.sample(frac=1)
print("Dataframe Shuffled...")
print(iris_frame.info())

# #save dataframe locally for troubleshooting
# iris_frame_shuffled.to_csv("data/iris.csv", sep=",", index=False)
# print("Dataframe saved locally...")

##profile
from evidently.model_profile import Profile 
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection

iris_data_drift_profile = Profile(sections=[DataDriftProfileSection()])
iris_data_drift_profile.calculate(iris_frame_shuffled[:125], iris_frame_shuffled[125:], column_mapping = None)
print("Dataset Profile:{}".format(type(iris_data_drift_profile.json()))) 

with open("reports/data_profile.json", "w", encoding="utf-8") as f:
	json.dump(json.loads(iris_data_drift_profile.json()), f, ensure_ascii=False, indent=4)

iris_data_drift_profile = Profile(sections=[DataDriftProfileSection(), CatTargetDriftProfileSection()])
iris_data_drift_profile.calculate(iris_frame_shuffled[:125], iris_frame_shuffled[125:], column_mapping = None)
print("Dataset Profile:{}".format(type(iris_data_drift_profile.json()))) 

with open("reports/data_target_profile.json", "w", encoding="utf-8") as f:
	json.dump(json.loads(iris_data_drift_profile.json()), f, ensure_ascii=False, indent=4)


## data quality dashboard
from evidently import ColumnMapping 
from evidently.dashboard.tabs import DataQualityTab
from evidently.model_profile.sections import DataQualityProfileSection
from sklearn.ensemble import RandomForestClassifier

ref_data = iris_frame_shuffled[:125] #baseline
#prod_data = iris_frame_shuffled[125:] #prod
prod_data = pd.read_csv("data/test.csv", sep=",")

#train a model
target = 'target'
numerical_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

features = numerical_features
model = RandomForestClassifier(random_state=0)
model.fit(ref_data[features], ref_data[target])

# print("Model Prediction Started")
# ref_data['prediction'] = model.predict(ref_data[features])
# prod_data['prediction'] = model.predict(prod_data[features])

# column_mapping = ColumnMapping(target, 
# 	'prediction',
# 	task='classification',
# 	numerical_features=numerical_features)

iris_data_drift_report = Dashboard(tabs=[DataQualityTab(), DataDriftTab(), CatTargetDriftTab()])
iris_data_drift_report.calculate(ref_data, prod_data, column_mapping = None)
iris_data_drift_report.save("reports/my_report_2.html")

profile = Profile(sections=[DataQualityProfileSection()])
profile.calculate(ref_data, prod_data, column_mapping=column_mapping)
profile_response = profile.json()
with open("reports/data_quality_profile.json", "w", encoding='utf-8') as f:
	json.dump(json.loads(profile_response), f, ensure_ascii=False, indent=4) 
