import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, auc
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV

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
		'num_leaves': 31,
		'learning_rate': 0.05,
		'feature_fraction': 0.9,
		'bagging_fraction': 0.8,
		'bagging_freq': 5,
		'verbose': 0
	}

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0, shuffle=True)
	# create dataset for lightgbm
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

	rf_clf_final = lgb.train(params,
					lgb_train,
					num_boost_round=20,
					valid_sets=lgb_eval,
					early_stopping_rounds=5)



	# cv_scores = cross_val_score(rf_clf_final, X, y, cv=5)
	# print('CV SCores = ',cv_scores)
	# print("Avg CV Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
	# predict
	y_pred = rf_clf_final.predict(X_test, num_iteration=rf_clf_final.best_iteration)
	# eval
	# print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)



	y_pred = rf_clf_final.predict(X_test)

	print 'predicted = \n', y_pred[10:20 ]

	print "test = \n", y_test.values[10:20 ]
	# score = rf_clf_final.score(X_test, y_test)
	# print "score = ", score

	# print " CM = \n", confusion_matrix(y_test.values, y_pred)

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
		p = model.predict(X_test)
		print p[:5]
		return p

	except Exception as e:
		print('Error in making prediction', e)
		return None

if __name__ == "__main__":
	input_dataframe = 'train_moreCat.csv'
	input_dataframe_test = 'test_moreCat.csv'

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

	# model = make_model(train.values, y)


	with open(input_dataframe_test, 'rb') as infile:
		test = pd.read_csv(infile)
		test.fillna('NA', inplace=True)
		if 'DG1' in list(test):
			test = test.drop('DG1', axis=1)
		print('test shape ', test.shape)

	dtrain = xgb.DMatrix(train.values, label = y )
	dtest = xgb.DMatrix(test.values)
	#
	# # specify parameters via map
	param = {'max_depth':4 , 'eta': 1, 'silent': 1, 'objective':'binary:logistic'}
	num_round = 100
	bst = xgb.train(param, dtrain, num_round)

	rng = np.random.RandomState(31337)

	# X = train.values
	# kf = KFold(n_splits=2, shuffle=True, random_state=rng)
	# for train_index, test_index in kf.split(X):
	# 	xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4).fit(X[train_index], y[train_index])
	# 	predictions = xgb_model.predict(X[test_index])
	# 	actuals = y[test_index]
	# 	print(confusion_matrix(actuals, predictions))
	# 	model_recall_score = recall_score(actuals, predictions, average='macro')
	# 	auc_score = roc_auc_score(actuals, predictions)
	# 	print("recall score", model_recall_score, " auc score ", auc_score)
	#
	# model = xgb.XGBClassifier(n_estimators=100, max_depth=4).fit(X,y)


	# importances = model.feature_importances_
	# indices = np.argsort(importances)[::-1]
	# feature_names = np.asarray(list(train))[indices]
	# importances = importances[indices]
	# feature_importance = DataFrame(
	# 	{'feature': feature_names, 'importance': np.round(importances, 3)})
	# feature_importance = feature_importance[feature_importance['importance'] > 0.004]

	fo = open('featmap2.txt', 'w')
	for i in range(len(list(train))):
		fo.write('%d\t%s\ti\n' % (i, list(train)[i]))
	# print("important features ", feature_importance)
	xgb.plot_tree(bst, fmap = 'featmap2.txt')
	# gph = xgb.to_graphviz(bst, fmap = 'featmap2.txt')
	# gph.save('graph.png')
	plt.show()
	# label = model.predict_proba(test.values)
	# #
	# output = []
	# for i in label:
	# 	output.append(i)
	#
	# index = range(0, len(label))
	# print(len(index))
	#
	# df = DataFrame({'test_id': index, 'is_female': output})
	# print(df.head())
	# df.to_csv(path_or_buf='xgboost_small_march1.csv', index=False)
