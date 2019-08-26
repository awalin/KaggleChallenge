import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, auc
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

from sklearn.metrics import matthews_corrcoef


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

	params = {
		'task': 'train',
		'boosting_type': 'gbdt',
		'metric': {'l2', 'auc','binary_logloss'},
		'objective': 'binary',
		'learning_rate': 0.01,
		'feature_fraction': 0.9,
		'bagging_fraction': 0.8,
		'bagging_freq': 5,
		'verbose': 0
	}

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0, shuffle=True)
	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	rf_clf_final = lgb.train(
					params,
					lgb_train,
					num_boost_round=50,
					valid_sets=lgb_eval,
					early_stopping_rounds=5)

	y_pred = rf_clf_final.predict(X_test)

	print 'predicted = \n', y_pred[10:20]


	output = []
	for i in y_pred:
		output.append(np.round(i))

	print "test = \n", y_test.values[10:20]



	# score = rf_clf_final.score(X_test, y_test)
	# print "score = ", score

	# print " CM = \n", confusion_matrix(y_test.values, y_pred)

	try:
		model_recall_score = recall_score(y_test, output, average='macro')
		auc_score = roc_auc_score(y_test, output)
		print("recall score", model_recall_score, " auc score ", auc_score)

		mcc = matthews_corrcoef(y_test, output)
		print "MCC score ", mcc
	except Exception as e:
		print("Error calculating recall score", e)

	return rf_clf_final


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
	input_dataframe = 'train_encoded_2.csv'
	input_dataframe_test = 'test_encoded_2.csv'

	with open(input_dataframe, 'rb') as infile:
		train = pd.read_csv(infile)
		print('train small shape ', train.shape)
		train.fillna('NA', inplace=True)
		# train = train.drop('ANALWT_C', axis=1)
		# print train.head()
		# train['Criminal'] = train['Criminal'].astype(str)
		y = train['Criminal']
		#  stupid mistake, need to remove the y label before prediction
		train = train.drop('Criminal', axis=1)

	# model = make_model(train.values, y)


	with open(input_dataframe_test, 'rb') as infile:
		test = pd.read_csv(infile)
		test.fillna('NA', inplace=True)

		# test = test.drop('ANALWT_C', axis=1)
		print('test shape ', test.shape)

	dtrain = xgb.DMatrix(train.values, label = y )
	dtest = xgb.DMatrix(test.values)
	#
	# # specify parameters via map
	param = {'max_depth':4 , 'eta': 1, 'silent': 1, 'objective':'binary:logistic'}
	num_round = 100
	# bst = xgb.train(param, dtrain, num_round)

	rng = np.random.RandomState(31337)

	X = train.values

	# kf = KFold(n_splits=2, shuffle=True, random_state=rng)
	# for train_index, test_index in kf.split(X):
	# 	xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4).fit(X[train_index], y[train_index])
	# 	predictions = xgb_model.predict(X[test_index])
	# 	actuals = y[test_index]
	# 	print(confusion_matrix(actuals, predictions))
	# 	model_recall_score = recall_score(actuals, predictions, average='macro')
	# 	auc_score = roc_auc_score(actuals, predictions)
	# 	print("recall score", model_recall_score, " auc score ", auc_score)


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0, shuffle=True)

	model = xgb.XGBClassifier(n_estimators=100, max_depth=5).fit(X_train,y_train)

	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	feature_names = np.asarray(list(train))[indices]
	#
	# print feature_names
	importances = importances[indices]
	feature_importance = DataFrame(
		{'feature': feature_names, 'importance': np.round(importances, 3)})
	feature_importance = feature_importance[feature_importance['importance'] > 0.004]

	print feature_importance.head()

	y_pred = model.predict(X_test)
	y_true = y_test

	print y_pred[:5]
	print y_test.values[:5]

	mcc = matthews_corrcoef(y_true.values, y_pred)
	print "MCC ", mcc

	label = make_prediction(test.values, model)

	output = []
	for i in label:
		output.append(int(np.round(i)))

	id_field = 'PERID'

	with open('criminal_test.csv', 'rb') as infile:
		sub = pd.read_csv(infile)
		id_field = 'PERID'
		ids = sub[id_field]

	df = DataFrame()
	df[id_field] = sub[id_field]
	df['Criminal'] = output
	print(df.head())
	df.to_csv(path_or_buf='submission_crime_xgb_depth_5.csv', index=False)
