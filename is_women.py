import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
import numpy as np
from functools import *
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, auc
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV


import seaborn as sns # data visualization library
import matplotlib.pyplot as plt

tuned_parameters = {'n_estimators': [200, 250, 300],
					 'max_depth': [20, 30, 40, 50]}
					# 'learning_rate':[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
					# }
#  add learning_rate : float, optional (default=0.1) for GBT

scores = ['roc_auc']

scaled_columns = ['MM17_', 'LN2_', 'IFI4_', 'IFI17_', 'LN1A', 'LN1B', 'LN2_', 'MM32_', 'MM5A_', 'MM8_', 'MMP2_',
				  'MMP4_', 'MT14A_', 'MT14C_', 'MT17_', 'FF16_', 'FF9',  'IFI14_',
				  'IFI15_', 'IFI2_', 'IFI4_', 'MM42_', 'MM5_', 'MM8_','DL8', 'DG1', 'MM32_','MM31_']

categorical_columns = ['AA3', 'AA5', 'AA6', 'AA8', 'AB3', 'AB6', 'DG3', 'DG3A', 'DG4', 'DG6', 'DG14', 'DL1', 'FL11',
					   'FL12','FL13','FL14','FL15','FL16','FL17','FL18','DL2', 'DL12', 'DL13', 'DL15', 'DL24', 'DL27',
					   'DL28', 'DL5', 'FB18', 'FB19', 'FB2', 'FB20', 'FB21', 'FB24', 'FB25', 'FB28_', 'FF13', 'FF2',
					   'FF2A', 'FF3', 'FF5', 'FF6_', 'FL1', 'FL10','RI7_',
					   'FL12', 'FL13', 'FL14', 'FL15', 'FL16', 'FL17', 'FL18', 'FL2', 'FL3', 'FL4', 'FL9A', 'FL9B',
					   'FL9C', 'G2P2_', 'GN1', 'GN2', 'GN3', 'GN4', 'GN5', 'IFI16_',
					   'IFI21', 'IFI24', 'IFI5_', 'MM11_', 'MM10B', 'MM12', 'MM13', 'MM14', 'MM20', 'MM21', 'MM25',
					   'MM28', 'MM29', 'MM30', 'MM34', 'MM7_','RI5_','MMP3_', 'MT13_', 'MT11', 'MT14_', 'MT14B',
					   'RI6_', 'MT9', 'MT7A', 'MT5', 'MT6', 'MT6A', 'MT6B', 'MT1A', ]

boolean_columns = ['AA19', 'AA20', 'AA21', 'DG12B_', 'DG5_', 'DL16', 'DL17', 'DL18', 'DL19', 'DL20', 'DL21', 'DL22', 'DL0',
				   'DL23', 'DL25_', 'DL4_', 'DL3', 'FB1_', 'DL6', 'DL7', 'FB16_', 'FB16A_',
				   'FB17_', 'FB22_', 'FB23_', 'FB26_', 'FB27_', 'FB29_', 'FB4_', 'FB10_', 'FF1', 'FB3', 'FF14_',
				   'FF19_', 'FF20', 'FF4', 'FL6_', 'IFI1__', 'IFI11_', 'IFI10_', 'IFI11_', 'IFI12_', 'IFI20_', 'IFI22_',
				   'IFI3_', 'MM10A_', 'MM15_', 'MM2_', 'MM24_', 'MM3_', 'MM33', 'MM36_', 'MM37_', 'MM38_',
				   'MM4_', 'MM40_', 'MM6_', 'MMP1_', 'MT16_', 'MT18_', 'MT4_', 'MT2', 'MT7',
				   'MT8', 'RI8_', 'DL10', 'MM36_','MM37_','MM35']

maybe_girl = ['G2P1_3', 'G2P1_11', 'G2P1_15', 'G2P1_16', 'G2P1_5', 'G2P1_6']
#  DG1 = age
to_delete = ['AA4', 'AA7', 'AA14','AA15','MT12','MT6C','DG1']
check_and_delete = ['AA','G2P3_', 'FF7_', 'MT3_', '_OTHERS', 'MT12_', 'DL1_OTHERS']

important_features = ['DG6=2', 'DL0', 'DL1=7', 'MT1A=2.0', 'DG6=1', 'DL3', 'MT1A=1.0',
       'MT6=1.0', 'FL4=2', 'DL2=nan', 'DG3=6', 'MT6=2.0', 'GN1=1.0', 'DG1',
       'FL4=1', 'MT10=2', 'MT4_1', 'GN5=1', 'MT14C_2', 'GN4=1', 'MT14C_3',
       'DG3=3', 'GN3=1', 'MT4_3', 'MT4_6', 'GN3=2', 'GN5=2', 'MT6B=nan',
       'DL1=1', 'MT2', 'MT4_2', 'MT14C_1', 'GN1=nan', 'DG3=1', 'GN4=2',
       'MT6B=1.0', 'MT10=1', 'GN2=2', 'MT4_4', 'DG6=3', 'MT6A=nan',
       'DG8a=1', 'DG5_4', 'MT7', 'IFI14_2', 'MT6A=1.0', 'GN1=2.0',
       'MT14C_4', 'IFI15_2', 'MT4_5', 'IFI14_1', 'GN2=1', 'DG4=1',
       'DL2=1.0', 'IFI15_1', 'MT5=1.0', 'IFI17_2', 'DL5=5.0', 'LN1A',
       'LN2_2', 'LN2_1', 'MT18_5', 'LN1B', 'MT18_2', 'LN2_3', 'IFI17_1',
       'LN2_4', 'GN4=4', 'MT1A=3.0', 'DL8', 'DL1=9', 'FF9', 'DL7', 'DL1=8',
       'MT18_3', 'GN5=4', 'MT17_5', 'MT15=2.0', 'DG8a=2', 'GN2=4',
       'G2P1_11=1.0', 'MT17_2', 'DL4_5', 'IFI15_3', 'DG5_6', 'IFI14_3',
       'MT14A_2', 'DL14=1', 'G2P5_11=2.0', 'MT17_4', 'DL5=6.0', 'GN3=4',
       'MT16_4', 'DG10c', 'DL15=4', 'FF16_2', 'DL4_6', 'DL1=4', 'MT17_3',
       'DL1=99', 'FL4=8', 'MT17_1', 'AA3=3', 'MT7A', 'DG12C_1', 'DG8c=0',
       'DG9a=1.0', 'MT18_4', 'GN1=4.0', 'DG10b', 'FF16_1', 'MT17_12',
       'MT17_9', 'MT17_11', 'IFI17_3', 'DG12C_2', 'AA3=1', 'MT1=1',
       'DG11c', 'DL15=1', 'DG11b', 'DL23', 'MT17_8', 'DL15=3', 'IFI15_4',
       'DL4_99', 'MT14_2=1.0', 'FB2=2', 'FB4_1', 'FB20=15.0', 'MT18_1',
       'DL21', 'DL6', 'FF13', 'FL11=99', 'DG12B_1', 'FF3=99.0', 'IFI14_4',
       'DL19', 'FB26_1', 'FL4=99', 'DL24=99', 'DG5_7', 'IFI14_7', 'DL15=2',
       'MT17_6', 'FL8_2=4', 'DL2=30.0', 'DL22', 'FL9A=11', 'FB2=3',
       'AA3=4', 'FL16=2', 'DG9c=0.0', 'DL2=2.0', 'FL8_4=4', 'FL1=4',
       'IFI15_7', 'MT17_7', 'DG12B_2', 'FL15=2', 'FL10=99', 'FB4_2',
       'IFI14_5', 'DL1=10', 'DL16', 'DL11=99', 'DG9a=2.0', 'MT8', 'DL14=2',
       'FL8_5=4', 'IFI15_5', 'FL8_1=2', 'FL14=99', 'DG8b=0', 'MT16_96',
       'DL18', 'FL6_1', 'DL1=6', 'DL12=11.0', 'FL18=99', 'FL8_5=3',
       'DL14=4', 'DL11=0', 'FB4_4', 'MT18A_2=2.0', 'DL24=2', 'FB1_1',
       'FL8_2=3', 'FL14=1', 'IFI16_2=2.0', 'DL17', 'DL20', 'FB26_6',
       'MT1A=4.0', 'GN5=3', 'FL1=2', 'AA6=6.0', 'FL8_6=4', 'FL9A=1',
       'FB23_1', 'FL8_4=3', 'DL24=3', 'FL8_1=4', 'DG5_2', 'DL26_12=1',
       'FL2=2.0', 'FB26_99', 'FL16=1', 'FL15=99', 'DG8a=3', 'FL13=1',
       'FB13=99', 'DL26_12=2', 'FL8_7=3', 'FL17=1', 'DL14=3', 'FL17=99',
       'MT16_99', 'FL13=99', 'AA6=7.0', 'FL16=99', 'FB13=0', 'FF14_1',
       'IFI18=99', 'FL18=1', 'GN3=3', 'FB1_3', 'DL26_99=1', 'FL8_6=3',
       'G2P4_11=1.0', 'IFI16_1=1.0', 'IFI18=0', 'DG9b=0.0', 'FL17=2',
       'AA3=2', 'FL11=2', 'DG4=6', 'FB4_3', 'GN4=3', 'FL8_5=2', 'FL8_2=2',
       'FL6_3', 'MT17_10', 'DG4=5', 'FL8_3=3', 'FB2=1', 'DG5_5', 'AA6=8.0',
       'MT1=2', 'FL8_7=2', 'FL8_4=2', 'FL8_6=2', 'FB24=15.0', 'GN1=3.0',
       'DG8a=4', 'DL26_99=2', 'DL1=5', 'FB26_10', 'FB26_8', 'FF2=1.0',
       'FL8_1=3', 'FL15=3', 'FB26_2', 'DG5_1', 'IFI14_6', 'FL8_3=4',
       'FL8_7=4', 'FL8_3=2', 'IFI16_1=2.0', 'FB18=5', 'DG9a=0.0',
       'IFI16_2=1.0', 'FB26_5', 'GN2=3', 'MT15=1.0', 'FB1_2', 'G2P1_9=2.0',
       'FL9B=11.0', 'DG3A=4', 'FL10=11', 'FB20=1.0', 'MT18A_4=2.0',
       'DG8b=1', 'FB19=10', 'DL25_1', 'FB26_11', 'FL6_2', 'IFI15_6',
       'MM1=1', 'DG8c=1', 'FF2=2.0', 'DL25_3', 'FF10_2', 'FF14_2',
       'FB26_4', 'G2P1_11=2.0', 'FF5=3.0', 'FL11=1', 'DG4=7', 'FL7_2=2',
       'IFI24=2.0', 'FB26_7', 'DL24=1', 'FL11=3', 'DL25_2', 'FF10_1',
       'DL14=5', 'MT16_1', 'MT17_13', 'IFI17_7', 'FF6_7=2.0', 'FL6_4',
       'FL7_1=2', 'FB26_3', 'IFI17_5', 'DG13_2', 'FL3=3.0', 'FF6_1=2.0',
       'FB19B_1=2.0', 'FB3', 'DG5_10', 'IFI17_4', 'FL14=2', 'FL1=1',
       'MT14A_7', 'FB19B_4=2.0', 'MT18_8', 'MT1=0', 'DG13_7', 'FL8_1=1',
       'GN5=99', 'AA5=5.0', 'MT1A=99.0', 'FF6_3=2.0', 'DG6=7', 'FB20=2.0',
       'FB20=14.0', 'MT16_2', 'MT16_3', 'IFI20_9', 'DG9a=3.0', 'MT1=99',
       'FF6_6=2.0', 'FF6_2=2.0', 'FF2A=1.0', 'DL24=5', 'DG3A=2',
       'FB19B_4=99.0', 'FF6_9=2.0', 'FF5=1.0', 'MT9=15.0', 'FB19B_3=99.0',
       'IFI16_1=3.0', 'FL15=1', 'FB19B_3=2.0', 'FL2=1.0', 'FB19B_2=2.0',
       'FB4_96', 'FF4', 'FB19B_2=99.0', 'FF6_10=2.0', 'FB19B_96=99.0',
       'MT18A_1=2.0', 'GN4=99', 'FB16A_8', 'IFI24=4.0', 'AA5=3.0',
       'FF6_4=2.0', 'FB19B_1=99.0', 'MT18_6', 'FL9B=6.0', 'DG3=5',
       'FL9A=3', 'DG3=99', 'IFI24=10.0', 'FB18=1', 'MT18_96', 'DG13_5',
       'G2P1_12=2.0', 'DL24=9', 'FL8_4=5', 'FL2=3.0', 'DL4_2', 'FF6_8=2.0',
       'FL7_1=99', 'MT6=3.0', 'FF19_4', 'DG6=99', 'FL7_4=2', 'G2P1_13=2.0',
       'MT14A_11', 'FL9C=11.0', 'FB19B_5=99.0', 'FL11=4', 'FF6_2=99.0',
       'FL10=1', 'DG8a=99', 'FB17_8', 'FL9B=2.0', 'IFI22_1', 'IFI16_2=3.0',
       'G2P1_1=2.0', 'G2P1_7=2.0', 'DG9b=1.0', 'FF6_5=2.0', 'DL26_5=2',
       'DG8a=5', 'FL8_3=5', 'FL3=8.0', 'FB19B_96=2.0', 'DL26_5=1', 'DG4=4',
       'IFI22_7', 'FL7_3=2', 'G2P1_3=2.0', 'FF19_1', 'FL3=2.0', 'DL25_4',
       'FL8_7=5', 'FL10=2', 'FL1=3', 'G2P1_8=2.0', 'MM1=2', 'FB26_9',
       'DL1=2', 'FL7_5=2', 'FB16A_1', 'G2P1_4=2.0', 'DG13_96', 'FF3=2.0',
       'FB19B_5=2.0', 'FB26_96', 'FL7_4=99', 'FL8_7=1', 'DG13_1',
       'IFI20_4', 'DL5=2.0', 'G2P1_99=2.0', 'FL7_6=99', 'FL12=1',
       'FL7_3=99', 'FF2A=2.0', 'G2P1_14=2.0', 'G2P1_6=2.0', 'FF6_7=99.0',
       'FL9C=6.0', 'FF6_9=99.0', 'DL14=6', 'DG4=99', 'DL24=4', 'IFI20_5',
       'G2P1_10=2.0', 'DG8b=2', 'FL9A=7', 'DG13_3', 'FL8_6=5', 'FL8_2=5',
       'FF6_10=99.0', 'IFI24=6.0', 'DL27=5.0', 'GN3=99', 'DL4_17',
       'FB24=1.0', 'GN2=99', 'FB17_1', 'DL25_5', 'FL7_2=99', 'FB19=1',
       'FL9B=7.0', 'FL8_3=1', 'FL8_5=1', 'DG4=3', 'FF6_8=99.0', 'DG14',
       'DL25_7', 'G2P1_5=2.0', 'FF6_2=1.0', 'FB20=6.0', 'DL1=3', 'FL12=99',
       'IFI16_2=7.0', 'FL8_5=5', 'FF6_6=99.0', 'MT6=4.0', 'FF6_8=1.0',
       'FL9B=1.0', 'FL7_6=2', 'IFI16_2=99.0', 'FL9B=3.0', 'FB22_8',
       'DL25_6', 'G2P1_96=2.0', 'MT18A_1=1.0', 'FF6_3=99.0', 'FL3=4.0',
       'G2P1_2=2.0', 'IFI24=1.0', 'DG9c=1.0', 'FF2A=12.0']

remove_fatures = []

def remove_nan(artifact):
	if pd.isnull(artifact):
		return 'NA'
	else:
		return artifact


def sorted_feature_importances(model, vectorizer):
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	feature_names = np.asarray(vectorizer.get_feature_names())[indices]
	importances = importances[indices]
	feature_importance = DataFrame(
		{'feature': feature_names[0:500], 'importance': np.round(importances[0:500], 3)})
	feature_importance = feature_importance[feature_importance['importance'] > 0.002]
	print("important features ",feature_importance)

	# Plot the feature importances of the forest
	# ranges  = 20

	# plt.figure(1, figsize=(10, 8))
	# plt.title("Feature importances")
	# plt.bar(range(ranges), importances[:20],
	# 		color="g", align="center")
	# plt.xticks(range(ranges), feature_names[:20], rotation=90)
	# plt.xlim([-1,  ranges])
	# plt.show()


	return feature_importance

# a lot of very null columns happen to have great value
def is_many_null(df, column_name):
	count_nan = len(df[column_name]) - df[column_name].count()
	null_perct = 100.00 * (count_nan * 1.0 / len(df[column_name]))
	return null_perct

def column_vals(df, col):
	res = df[col].value_counts()
	print('Column        =', col)
	print('# uniq values =', len(res))
	print()
	print(res)
	return res

def convert_age(field):
	try:
		if field is not None:
			return int(2018-field)
	except:
		return 0
	return 0

def format_data(df0):
	# df = shuffle(df0, random_state=0)
	df = df0
	y = df['is_female']

	df = df.drop('is_female', axis=1)
	assert isinstance(df, DataFrame)

	if isinstance(df, dict):
		df_to_dict = df
	else:
		df_to_dict = df.to_dict(orient="records")

	vec = DictVectorizer(sparse=False)
	vec.fit(df_to_dict)
	X = vec.transform(df_to_dict)
	print('inside make model after one hot encoding= ', X.shape)
	columns_names = vec.feature_names_
	# print len(columns_names)
	# if you already know which columns to remove use this
	# for feat in columns_names:
	# 	if feat not in important_features:
	# 		print("dropping feature with less importance = ", feat)
	# 		input_dataframe = input_dataframe.drop(feat, axis=1)
	## Remove all minor features when you do not have teh list

	rf_clf = RandomForestClassifier(n_estimators=200, max_depth=35)
	rf_clf.fit(X, y)
	imp = rf_clf.feature_importances_
	# sorted_feature_importances(rf_clf, vec)

	threshold_for_features = 0.001
	input_dataframe = DataFrame(data=X, columns=columns_names)
	for index, value in enumerate(imp):
		if value <= threshold_for_features:
			# print("dropping feature with little significance, = ", columns_names[index])
			key = columns_names[index]
			input_dataframe = input_dataframe.drop(key, axis=1)

	s = set(important_features)
	temp3 = [x for x in list(input_dataframe) if x not in s]
	for feat in temp3:
		if feat.endswith("=NA") or feat.endswith("=nan"):
			# print("dropping feature with no value = ", feat)
			input_dataframe = input_dataframe.drop(feat, axis=1)

	# print('Columns to remove as they are less important')

	# clf_rf_4 = RandomForestClassifier()
	# rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='accuracy')  # 5-fold cross-validation
	# rfecv = rfecv.fit(input_dataframe.values, y)
	# print('Optimal number of features :', rfecv.n_features_)
	# print('Best features :', X.columns[rfecv.support_])

	# num_features = rfecv.n_features_
	# num_features = 100
	# find best scored num_features features
	# select_feature = SelectKBest(chi2, k=num_features).fit(input_dataframe, y)
	# input_X = select_feature.transform(input_dataframe)
	# scores = select_feature.scores_
	# col_names = input_dataframe.columns
	#
	# indices = np.argsort(scores)[::-1]
	# feature_names = np.asarray(col_names)[indices]
	# print scores[indices][:5]
	# print('Feature list:', feature_names[0:5])
	# input_dataframe = DataFrame(data=input_X, columns=feature_names[0:num_features])

	# WE SHOULD STORE THIS DATA FRAME IN FILE SYSTEM
	# Saving this for reusing, this is just the one hot encoded data with important features

	df_to_dict = input_dataframe.to_dict(orient="records")
	vec = DictVectorizer(sparse=False)
	vec.fit(df_to_dict)

	print(" modified data frame ", input_dataframe.shape)
	with open('train_select_1.csv', 'wb') as infile:
		input_dataframe['is_female'] = y
		print("input df shape to csv ", input_dataframe.shape)
		input_dataframe.to_csv(infile, index=False)


	return input_dataframe, vec

def make_model(X, y):
	# Run some model selection GRID Search alg here, to find the best hyper params

	clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10)
	clf.fit(X, y)
	#
	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)

	rf_clf_final = clf.best_estimator_
	# rf_clf_final = GradientBoostingClassifier(max_depth=40, n_estimators=200, learning_rate=0.3)
	# rf_clf_final = RandomForestClassifier(n_estimators=300, class_weight='balanced', max_depth=40)

	cv_scores = cross_val_score(rf_clf_final, X, y, cv=10)
	print('CV SCores = ',cv_scores)
	print("Avg CV Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
	# rf_clf_final.fit(X_new, y)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
	# y_pred = rf_clf_final.predict(X_test)
	rf_clf_final.fit(X_train, y_train)
	score = rf_clf_final.score(X_test, y_test)
	print score
	#
	# try:
	# 	model_recall_score = recall_score(y_test, y_pred, average='macro')
	# 	auc_score = roc_auc_score(y_test, y_pred)
	# 	print("recall score", model_recall_score, " auc score ", auc_score, " test score ", score)
	# except:
	# 	print("Error calculating recall score")
	# feature_importances = sorted_feature_importances(rf_clf_final, vec)

	# print('important features ', feature_importances)
	return rf_clf_final


def maybe_female(df):
	if df['G2P1_3'] == 1 or df['G2P1_11'] == 1 or df['G2P1_15'] == 1 or df['G2P1_16'] == 1 or df['G2P1_5'] == 1 or df[
		'G2P1_6'] == 1:
		return 1
	else:
		return 0

def format_string(field):
	try:
		if field is not None:
			return ""+str(field)+""
	except:
		return "NA"
	return "NA"

# see if this person have any of this method used by them. total 12 methods
def makeG2P2_one(method, columns, df):
	used = 0
	for i in range(1, 17 + 1):
		name = "G2P2_" + str(i)
		if name in columns:
			# print(name, ' value = ' ,df[name])
			if df[name] == method:
				used = 1
				break

	if used == 1:
		return 1
	return 0

def format_number(field):
	try:
		if field is not None:
			if field == 99:
				return int(0)
			return int(field)
		else:
			return 0
	except:
		return 0
	return 0

def remove_nulled_columns():
	# for key in all_clumns:
	# 	perc = is_many_null(df, key)
	# 	if perc >= 70.00:
	# 		print(key , ' has ', perc, ' % null values, deleting that key')
	# 		to_delete.append(key)
	# 		df = df.drop(key, axis=1)
	return

def get_data(name):
	file = name + '.csv'
	df = pd.read_csv(file)
	id_field = name + '_id'
	df = df.drop(id_field, axis=1)
	print(name + '  file read ', df.shape)
	for method in range(1, 12 + 1):
		df["delivery_" + str(method)] = df.apply(partial(makeG2P2_one, method, df.columns), axis=1)

	for key in to_delete:
		if key in df.columns:
			df = df.drop(key, axis=1)

	numericals = scaled_columns+boolean_columns
	#  change birth year to age
	# df['DG1'] = df['DG1'].apply(convert_age)

	for key in list(df):
		is_num = False
		for col in numericals:
			#  substring check
			if col in key:
				#  if the name is a substring of the key name
				is_num = True
				break

		if is_num is True:
			df[key] = df[key].apply(format_number)
		else:
			# default is categorical
			df[key] = df[key].astype(str)
			df[key] = df[key].apply(format_string)

	# print(key, ' is numerical ')

	all_clumns = list(df.columns.values)

	df['receive_female_loan'] = df.apply(maybe_female, axis=1)

	if name is 'train':
		#  if training, make the list of useless columns
		for i in range(1, 17):
			del_key = "G2P2_" + str(i)
			to_delete.append(del_key)
			df = df.drop(del_key, axis=1)

		for key in all_clumns:
			# print(key)
			is_del = False
			for col in check_and_delete:
				# print('col to delete ', col )
				if col in key:
					#  if the name is a substring of the key name
					is_del = True
					break
			if is_del is True:
				to_delete.append(key)
				df = df.drop(key, axis=1)
				# print("deleting key = ", key)
	else:
		#  For test, just delete them because now we have the list of deletable columns
		for key in to_delete:
			if key in df.columns:
				df = df.drop(key, axis=1)

	df.fillna('NA', inplace=True)
	# print('To delete = ',to_delete)
	return df

def format_test(df, vec):

	df_to_dict = df.to_dict(orient="records")
	to_predict = vec.transform(df_to_dict)
	print("test features ", to_predict.shape, ' vec ', len(vec.feature_names_))
	with open('test_select_1.csv', 'wb') as infile:
		test_data_frame = DataFrame(data=to_predict, columns=vec.get_feature_names())
		print test_data_frame.head()
		test_data_frame.to_csv(infile, index=False)
	return to_predict


def make_prediction(X_test, model):
	try:
		print('inside make prediction')
		p = model.predict(X_test)
		return p

	except Exception as e:
		print('Error in making prediction', e)
		return None

if __name__ == "__main__":

	df = get_data('train')
	print("df shape ", df.shape)
	input_dataframe , vec = format_data(df)

	df_t = get_data('test')
	print("df test shape ", df_t.shape)
	X_test= format_test(df_t, vec)