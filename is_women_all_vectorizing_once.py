from sklearn.feature_extraction import DictVectorizer
from pandas import DataFrame
from functools import *
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import lightgbm as lgb


def combine_and_delete(df):
	df['if_from_town'] = np.where(df.AA5.isnull(), 0, 1)
	del df['DG12B_2']
	del df['DG12C_2']
	del df['DG13_7']
	del df['DG13_OTHERS']
	cols_to_combine = ['DG13', 'DL25', 'DL26', 'MT4']
	# xxxx = ['DG13']
	for col_to_combine in cols_to_combine:
		group_cols = [col for col in df if col.startswith(col_to_combine)]
		group_col_1 = group_cols[0]
		df[col_to_combine] = 0
		for group_col in group_cols:
			df[group_col] = df[group_col].replace(to_replace=2, value=0)
			df[col_to_combine] = df[col_to_combine] + df[group_col]
		if df[group_col_1].isnull().sum() > 0:
			new_col_name = 'if_' + col_to_combine + '_is_null'
			df[new_col_name] = np.where(df[col_to_combine].isnull(), 1, 0)


def test_feature_generation(df):
	df['Age'] = 2018 - df['DG1']
	del df['DG1']
	# positive
	# fraction
	df['if_GN1_by_spouse'] = np.where(df['GN1'] == 2, 1, 0)
	df['if_GN2_by_spouse'] = np.where(df['GN2'] == 2, 1, 0)
	df['if_GN3_by_spouse'] = np.where(df['GN3'] == 2, 1, 0)
	df['if_GN4_by_spouse'] = np.where(df['GN4'] == 2, 1, 0)
	df['if_GN5_by_spouse'] = np.where(df['GN5'] == 2, 1, 0)

	# medium
	df['if_low_education'] = np.where(df['DG4'] <= 2, 1, 0)
	df['if_other_main_income'] = np.where(df['DL0'] == 2, 1, 0)
	df['if_supported'] = np.where(df['DL5'] == 5, 1, 0)
	df['if_no_phone'] = np.where(df['MT2'] == 2, 1, 0)
	df['if_no_sim'] = np.where(df.MT15.isnull(), 0, 1)

	# high
	df['if_widow'] = np.where(df['DG3'] == 6, 1, 0)
	df['if_borrow_phone_from_spouse'] = np.where(df['MT7A'] == 1, 1, 0)
	# super
	df['if_spouse_househead'] = np.where(df['DG6'] == 2, 1, 0)
	df['if_housewife'] = np.where(df['DL1'] == 7, 1, 0)
	df['if_spouse_decide_phone'] = np.where(df['MT1A'] == 2, 1, 0)
	df['if_phone_bought_by_spouse'] = np.where(df['MT6'] == 2, 1, 0)

	# negative
	# fraction
	df['if_GN1_by_myself'] = np.where(df['GN1'] == 1, 1, 0)
	df['if_GN2_by_myself'] = np.where(df['GN2'] == 1, 1, 0)
	df['if_GN3_by_myself'] = np.where(df['GN3'] == 1, 1, 0)
	df['if_GN4_by_myself'] = np.where(df['GN4'] == 1, 1, 0)
	df['if_GN5_by_myself'] = np.where(df['GN5'] == 1, 1, 0)
	# low
	df['if_high_education'] = np.where(df['DG4'] >= 9, 1, 0)
	df['if_self_business'] = np.where((df['DL5'] == 6) | (df['DL5'] == 19) | (df['DL5'] == 20), 1, 0)
	df['if_have_phone'] = np.where(df['MT2'] == 1, 1, 0)
	# medium
	df['if_full_time'] = np.where((df['DL1'] == 1) | (df['DL1'] == 5), 1, 0)
	df['if_myself_decide_phone'] = np.where(df['MT1A'] == 1, 1, 0)
	# high
	df['if_myself_main_income'] = np.where(df['DL0'] == 1, 1, 0)
	df['if_phone_bought_by_myself'] = np.where(df['MT6'] == 1, 1, 0)
	df['if_myself_househead'] = np.where(df['DG6'] == 1, 1, 0)
	df['if_have_drive_license'] = np.where(df['DG5_4'] == 1, 1, 0)


sns.set(color_codes=True)

tuned_parameters = {'n_estimators': [200, 250, 300],
					 'max_depth': [20, 30, 40, 50]}
					# 'learning_rate':[0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
					# }
#  add learning_rate : float, optional (default=0.1) for GBT
#  TODO : vectorize test and train data together, because there are values in test data not present in train data
scores = ['roc_auc']

scaled_columns = ['MM17_', 'LN2_', 'IFI4_', 'IFI17_', 'LN1A', 'LN1B', 'LN2_', 'MM32_', 'MM5A_', 'MM8_', 'MMP2_',
				  'MMP4_', 'MT14A_', 'MT14C_', 'MT17_', 'FF16_', 'FF9',  'IFI14_', 'DG1',
				  'IFI15_', 'IFI2_', 'IFI4_', 'MM42_', 'MM5_', 'MM8_','DL8', 'DG1', 'MM32_','MM31_', '_id']

categorical_columns = ['AA3', 'AA5', 'AA6', 'AA8', 'AB3', 'AB6', 'DG3', 'DG3A', 'DG4', 'DG6', 'DG14', 'DL1', 'FL11',
					   'FL12','FL13','FL14','FL15','FL16','FL17','FL18','DL2', 'DL12', 'DL13', 'DL15', 'DL24', 'DL27',
					   'DL28', 'DL5', 'FB18', 'FB19', 'FB2', 'FB20', 'FB21', 'FB24', 'FB25', 'FB28_', 'FF13', 'FF2',
					   'FF2A', 'FF3', 'FF5', 'FF6_', 'FL1', 'FL10','RI7_','MT15',
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
       'MT6=1.0', 'FL4=2', 'DL2=nan', 'DG3=6', 'MT6=2.0', 'GN1=1.0',
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

def convert_land(field):
	try:
		if field is not None:
			return (int(field)>1)
	except:
		return 0
	return 0


def format_data(df0, df_ts):
	# df = shuffle(df0, random_state=0)
	df = df0
	train_size = df.shape[0]
	print df.head()
	y = df['is_female']

	df = df.drop('is_female', axis=1)
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
	# rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10)
	# rf_clf.fit(X[0:train_size], y)
	# imp = rf_clf.feature_importances_
	# threshold_for_features = 0.001
	# for index, value in enumerate(imp):
	# 	if value <= threshold_for_features:
	# 		key = columns_names[index]
	# 		input_dataframe = input_dataframe.drop(key, axis=1)
	# This part was about removing un important columns


	s = set(important_features)
	temp3 = [x for x in list(input_dataframe) if x not in s]
	for feat in temp3:
		if feat.endswith("=NA") or feat.endswith("=nan"):
			# print("dropping feature with no value = ", feat)
			input_dataframe = input_dataframe.drop(feat, axis=1)

	df_to_dict = input_dataframe.to_dict(orient="records")
	vec = DictVectorizer(sparse=False)
	vec.fit(df_to_dict)

	print(" modified data frame ", input_dataframe.shape)

	input_train_df = input_dataframe[0:train_size]
	input_test_df = input_dataframe[train_size:]

	with open('train_moreCat_2.csv', 'wb') as infile:
		input_train_df['is_female'] = y
		print("input df shape to csv ", input_train_df.shape)
		input_train_df.to_csv(infile, index=False)

	with open('test_moreCat_2.csv', 'wb') as infile:
		print("input df shape to csv ", input_test_df.shape)
		input_test_df.to_csv(infile, index=False)


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
			return int(field)
		else:
			return 99
	except:
		return 99
	return 99

def get_data():
	name = 'train'
	file = name + '.csv'
	df = pd.read_csv(file, low_memory=False)
	id_field = name + '_id'
	df = df.drop(id_field, axis=1)
	y = df['is_female']
	# df = df.drop('is_female', axis=1)
	train_size = df.shape[0]
	print "Total training data = ", train_size


	name= 'test'
	file = name + '.csv'
	df_t = pd.read_csv(file, low_memory=False)
	id_field = name + '_id'
	df_t = df_t.drop(id_field, axis=1)

	combined_df = df.append(df_t)

	# print(name + '  file read ', df.shape)
	for method in range(1, 12 + 1):
		combined_df["delivery_" + str(method)] = combined_df.apply(partial(makeG2P2_one, method, combined_df.columns), axis=1)

	numericals = scaled_columns
	cat = boolean_columns + categorical_columns
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
			combined_df[key] = combined_df[key].apply(format_number)
		else:
			# default is categorical
			combined_df[key] = combined_df[key].astype(str)
			combined_df[key] = combined_df[key].apply(format_string)

	for key in list(df):
		is_c = False
		for col in cat:
			#  substring check
			if col in key:
				#  if the name is a substring of the key name
				is_c = True
				break

		if is_c is True:
			combined_df[key] = combined_df[key].astype(str)
			combined_df[key] = combined_df[key].apply(format_string)

	# print(key, ' is numerical ')

	combined_df['receive_female_loan'] = combined_df.apply(maybe_female, axis=1)
	combined_df['DL8'] = combined_df['DL8'].apply(convert_land)

	for i in range(1, 17):
		del_key = "G2P2_" + str(i)
		combined_df = combined_df.drop(del_key, axis=1)

	all_clumns = list(combined_df.columns.values)

	for key in to_delete:
		if key in combined_df.columns:
			combined_df = combined_df.drop(key, axis=1)

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
			if key in combined_df.columns.values:
				combined_df = combined_df.drop(key, axis=1)

	combined_df.fillna('NA', inplace=True)
	df_train = combined_df[0:train_size]
	df_train['is_female'] = y

	print "Train shape", df_train.shape
	df_test = combined_df[train_size:]
	print "df test ", df_test.shape

	# print('To delete = ',to_delete)
	return df_train, df_test

if __name__ == "__main__":

	# read data
	train = pd.read_csv("train.csv")
	test = pd.read_csv('test.csv')
	s1 = test['test_id']

	# basic prepossessing
	train["row_nulls"] = train.isnull().sum(axis=1)
	test["row_nulls"] = test.isnull().sum(axis=1)
	combine_and_delete(train)
	combine_and_delete(test)
	test_feature_generation(train)
	test_feature_generation(test)
	train = train[train.columns[train.isnull().mean() < 0.7]]

	# prepare for train and test
	col_names = list(train)
	for col in col_names:
		if train[col].dtypes == 'object':
			del train[col]
			continue
		train[col] = train[col].astype('float32', copy=False)
	train['is_female'] = train['is_female'].astype('int64')
	tempX = train

	print tempX.head()

	print tempX.shape

	print list(tempX)

	target = 'is_female'

	del tempX['train_id']

	predictors = [x for x in tempX.columns if x not in [target]]

	col_names_train = list(tempX)
	col_names_test = list(test)
	for col in col_names_test:
		if col not in col_names_train or test[col].dtypes == 'object':
			del test[col]
	for col in list(test):
		test[col] = test[col].astype('float32', copy=False)

	#####################################################################################

	#########################################################################################
	# train and predict with cv-auc = 0.9730*

	Y = train['is_female']

	final_params = {
		'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed': 27,
		'num_leaves': 32, 'learning_rate': 0.01, 'max_depth': -1, 'metric': 'auc',
		'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'bagging_seed': 72,
		'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
		'min_split_gain': 0.05, 'min_child_weight': 1, 'min_child_samples': 20, 'scale_pos_weight': 0.862}
	num_round = 2775
	train_data = lgb.Dataset(tempX[predictors], Y)
	lgb_model = lgb.train(final_params, train_data, num_round)

	ypred2 = lgb_model.predict(test)
	s2 = pd.Series(ypred2, name='is_female')
	out_df = pd.concat([s1, s2], axis=1).reset_index()
	del out_df['index']
	out_df.to_csv('lgb_seed27_simplified_fe.csv', index=False)

	#########################################################################################
	# train and predict with cv-auc = 0.972988
	final_params = {
		'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'seed': 12,
		'num_leaves': 49, 'learning_rate': 0.005, 'max_depth': -1, 'metric': 'auc', 'gamma': 8.3548,
		'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'bagging_seed': 22,
		'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.2970, 'reg_lambda': 0.0614,
		'min_split_gain': 0.1336, 'min_child_weight': 1, 'min_child_samples': 29, 'scale_pos_weight': 0.862}

	num_round = 3825

	train_data = lgb.Dataset(tempX[predictors], Y)
	lgb_model = lgb.train(final_params, train_data, num_round)

	ypred2 = lgb_model.predict(test)
	s2 = pd.Series(ypred2, name='is_female')
	out_df = pd.concat([s1, s2], axis=1).reset_index()

	del out_df['index']

	out_df.to_csv('lgb_seed12_simplified_fe.csv', index=False)


	# df_tr, df_ts = get_data()
	# print("df train shape ", df_tr.shape)
	# format_data(df_tr, df_ts)