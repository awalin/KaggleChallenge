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
#
tuned_parameters = {
	'n_estimators': [100, 150, 200, 250],
					 'max_depth': [10, 15, 20, 30, 40, 45, 50, 60]}
					# }
#  add learning_rate : float, optional (default=0.1) for GBT

scores = ['roc_auc']

def sorted_feature_importances(model, vectorizer):
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	feature_names = np.asarray(vectorizer.get_feature_names())[indices]
	importances = importances[indices]
	feature_importance = DataFrame(
		{'feature': feature_names[0:500], 'importance': np.round(importances[0:500], 3)})
	feature_importance = feature_importance[feature_importance['importance'] > 0.002]
	print("important features ",feature_importance)

	return feature_importance


def make_model(X, y):
	# Run some model selection GRID Search alg here, to find the best hyper params
	clf = GridSearchCV(RandomForestClassifier( class_weight='balanced', n_jobs= -1), tuned_parameters, cv=5)
	clf.fit(X, y)
	print("Best parameters set found on development set:\n")
	print(clf.best_params_)
	rf_clf_final = clf.best_estimator_
	# rf_clf_final = RandomForestClassifier(n_estimators=100,
	# 									  class_weight='balanced',
	# 									  max_depth=50,
	# 									  max_features=40)

	cv_scores = cross_val_score(rf_clf_final, X, y, cv=5)
	print('CV SCores = ',cv_scores)
	print("Avg CV Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, shuffle=True)
	rf_clf_final.fit(X_train, y_train)

	y_pred = rf_clf_final.predict(X_test)

	print 'predicted = \n', y_pred[10:20 ]

	print "test = \n", y_test.values[10:20 ]
	score = rf_clf_final.score(X_test, y_test)
	print "score = ", score

	print " CM = \n", confusion_matrix(y_test.values, y_pred)

	try:
		model_recall_score = recall_score(y_test, y_pred, average='macro')
		auc_score = roc_auc_score(y_test, y_pred)
		print("recall score", model_recall_score, " auc score ", auc_score)
	except Exception as e:
		print("Error calculating recall score", e)

	return rf_clf_final


def make_prediction(X_test, model):
	try:
		print('inside make prediction')
		p = model.predict_proba(X_test)
		print p[:5]
		return p

	except Exception as e:
		print('Error in making prediction', e)
		return None

if __name__ == "__main__":
	input_dataframe = 'train_combined_1_small.csv'
	input_dataframe_test = 'test_combined_1_small.csv'

	with open(input_dataframe, 'rb') as infile:
		train = pd.read_csv(infile)
		print('train small shape ', train.shape)
		train.fillna('NA', inplace=True)
		# train['is_female'] = train['is_female'].astype(str)

		y = train['is_female']
		if 'DG1' in list(train):
			print "dropping Age"
			train = train.drop('DG1', axis=1)
		#  stupid mistake, need to remove the y label before prediction
		train = train.drop('is_female', axis=1)

	model = make_model(train.values, y)

	with open(input_dataframe_test, 'rb') as infile:
		test = pd.read_csv(infile)
		test.fillna('NA', inplace=True)
		if 'DG1' in list(test):
			test = test.drop('DG1', axis=1)
		print('test shape ', test.shape)

	label = make_prediction(test.values, model)

	output = []
	for i in label:
		output.append(i)

	index = range(0, len(label))
	print(len(index))

	df = DataFrame({'test_id': index, 'is_female': output})
	print(df.head())
	df.to_csv(path_or_buf='submission_small_train_feb27_5.csv', index=False)
