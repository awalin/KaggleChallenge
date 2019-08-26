import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
import numpy as np
from functools import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, auc
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import matthews_corrcoef


tuned_parameters = {
	'n_estimators': [100, 150, 200, 250],
					 'max_depth': [10, 15, 20, 30, 40, 45, 50, 60]}

def sorted_feature_importances(model, columns):
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	feature_names = np.asarray(columns)[indices]
	importances = importances[indices]
	feature_importance = DataFrame(
		{'feature': feature_names[0:500], 'importance': np.round(importances[0:500], 3)})
	feature_importance = feature_importance[feature_importance['importance'] > 0.006]
	print("important features ",feature_importance)

	return feature_importance

def make_model(X, y):
	# Run some model selection GRID Search alg here, to find the best hyper params
	clf = RandomForestClassifier( class_weight='balanced',
								  n_jobs= -1,
								  max_depth=35,
								  n_estimators=150)
	# clf.fit(X, y)
	print("Best parameters set found on development set:\n")

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.150, random_state=0, shuffle=True)
	clf.fit(X_train, y_train)

	y_pred = clf.predict(X_test)

	print 'predicted = \n', y_pred[20:25 ]

	print "test = \n", y_test.values[20:25 ]
	score = clf.score(X_test, y_test)
	print "score = ", score

	print " CM = \n", confusion_matrix(y_test.values, y_pred)

	try:
		model_recall_score = recall_score(y_test, y_pred, average='macro')
		auc_score = roc_auc_score(y_test, y_pred)

		mcc = matthews_corrcoef(y_test.values, y_pred)
		print("recall score", model_recall_score, " auc score ", auc_score, " MCC Score ", mcc)
	except Exception as e:
		print("Error calculating recall score", e)

	return clf


def make_prediction(X_test, model):
	try:
		print('inside make prediction')
		p = model.predict(X_test)
		print p[:5]
		return p

	except Exception as e:
		print('Error in making prediction', e)
		return None

if __name__ == "__main__":
	input_dataframe = 'train_encoded.csv'
	input_dataframe_test = 'test_encoded.csv'

	with open(input_dataframe, 'rb') as infile:
		train = pd.read_csv(infile)
		print('train small shape ', train.shape)
		train.fillna('NA', inplace=True)

		y = train['Criminal']
		#  stupid mistake, need to remove the y label before prediction
		train = train.drop('Criminal', axis=1)

	model = make_model(train.values, y)

	sorted_feature_importances(model, list(train))


	with open(input_dataframe_test, 'rb') as infile:
		test = pd.read_csv(infile)
		test.fillna('NA', inplace=True)
		print('test shape ', test.shape)


	label = make_prediction(test.values, model)

	output = []
	for i in label:
		output.append(i)

	id_field = 'PERID'

	with open('criminal_test.csv', 'rb') as infile:
		sub = pd.read_csv(infile)
		id_field = 'PERID'
		ids = sub[id_field]

	df = DataFrame( {'Criminal': output})
	df[id_field] = sub[id_field]
	print(df.head())
	df.to_csv(path_or_buf='submission_crime_rf_4.csv', index=False)
