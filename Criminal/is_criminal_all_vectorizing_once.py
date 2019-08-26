from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
import numpy as np
from functools import *
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(color_codes=True)

tuned_parameters = {'n_estimators': [200, 250, 300],
					'max_depth': [20, 30, 40, 50]}
# 'learning_rate':[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
# }
#  add learning_rate : float, optional (default=0.1) for GBT
#  TODO : vectorize test and train data together, because there are values in test data not present in train data
scores = ['roc_auc']

to_delete_ = ['ANALWT_C', 'combined_df']
numerical = ['NRCH17_2', 'IRHH65_2','HLCNOTMO','HLCLAST','IRWELMOS','IIWELMOS','IRPINC3','IRFAMIN3']

remove_fatures = []


def remove_nan(artifact):
	if pd.isnull(artifact):
		return 'NA'
	else:
		return artifact


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


def format_data(df0, df_ts):
	# df = shuffle(df0, random_state=0)
	df = df0
	train_size = df.shape[0]
	print df.head()
	y = df['Criminal']

	df = df.drop('Criminal', axis=1)
	assert isinstance(df, DataFrame)

	df_combined = df.append(df_ts)
	df_combined.fillna('NA', inplace=True)

	if isinstance(df_combined, dict):
		df_to_dict = df_combined
	else:
		df_to_dict = df_combined.to_dict(orient="records")

	vec = DictVectorizer(sparse=False)
	vec.fit(df_to_dict)

	X = vec.transform(df_to_dict)
	print('inside make model after one hot encoding= ', X.shape)
	columns_names = vec.feature_names_
	input_dataframe = DataFrame(data=X, columns=columns_names)

	# This part is removing un important columns
	rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10)
	rf_clf.fit(X[0:train_size], y)
	imp = rf_clf.feature_importances_
	threshold_for_features = 0.001
	for index, value in enumerate(imp):
		if value <= threshold_for_features:
			key = columns_names[index]
			input_dataframe = input_dataframe.drop(key, axis=1)

	temp3 = list(input_dataframe)
	for feat in temp3:
		if feat.endswith("=NA") or feat.endswith("=nan") or feat.endswith("=99"):
			# print("dropping feature with no value = ", feat)
			input_dataframe = input_dataframe.drop(feat, axis=1)

	# This part was about removing un important columns

	df_to_dict = input_dataframe.to_dict(orient="records")
	vec = DictVectorizer(sparse=False)
	vec.fit(df_to_dict)

	print(" modified data frame ", input_dataframe.shape)

	input_train_df = input_dataframe[0:train_size]
	input_test_df = input_dataframe[train_size:]

	with open('train_encoded_2.csv', 'wb') as infile:
		input_train_df['Criminal'] = y
		print("input df shape to csv ", input_train_df.shape)
		input_train_df.to_csv(infile, index=False)

	with open('test_encoded_2.csv', 'wb') as infile:
		print("input df shape to csv ", input_test_df.shape)
		input_test_df.to_csv(infile, index=False)


def format_string(field):
	try:
		if field is not None:
			return "" + str(field) + ""
	except:
		return "NA"
	return "NA"


def format_number(field):
	try:
		if field is not None:
			return int(field)
		else:
			return -1
	except:
		return -1
	return -1


def get_data():
	name = 'criminal_train'
	file = name + '.csv'
	df = pd.read_csv(file, low_memory=False)
	id_field = 'PERID'
	df = df.drop(id_field, axis=1)
	y = df['Criminal']
	train_size = df.shape[0]
	print "Total training data = ", train_size
	name = 'criminal_test'
	file = name + '.csv'
	df_t = pd.read_csv(file, low_memory=False)
	id_field = 'PERID'
	df_t = df_t.drop(id_field, axis=1)

	combined_df = df.append(df_t)

	for key in list(df):
		if key in numerical:
			combined_df[key] = combined_df[key].apply(format_number)
		else:
			combined_df[key] = combined_df[key].astype(str)
			combined_df[key] = combined_df[key].apply(format_string)


	combined_df = combined_df.drop('ANALWT_C', axis=1)
	combined_df = combined_df.drop('VESTR', axis=1)

	combined_df.fillna('NA', inplace=True)
	df_train = combined_df[0:train_size]
	df_train['Criminal'] = y

	print "Train shape", df_train.shape
	df_test = combined_df[train_size:]
	print "df test ", df_test.shape

	# print('To delete = ',to_delete)
	return df_train, df_test


if __name__ == "__main__":
	df_tr, df_ts = get_data()
	print("df train shape ", df_tr.shape)
	format_data(df_tr, df_ts)
