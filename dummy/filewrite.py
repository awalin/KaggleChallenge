import math, os
import pandas as pd
from functools import *
# from predict_app.helper_classes.clean_up_data import column_vals
import cPickle
# from predict_app.helper_classes.virus_total_client import get_virus_total_client



from sqlalchemy import create_engine
from pandas import DataFrame
from datetime import datetime, timedelta


### Global Variables ##################################################################################################
#######################################################################################################################
importance_process_file = 'important_processes.pkl'
ext_file = 'common_extensions.pkl'

# client = get_virus_total_client()

####################################################################################################################################################################################################################################################
# declare lists

##########################################################################################################################

# textatlowestoffset analytics

def MZ(string):
	if string is None:
		return 0
	elif 'MZ' in string[:6]:
		return 1
	else:
		return 0


##########################################################################################################################
def system_PID(pid):
	if pid == '4':
		return 1
	else:
		return 0


def pull_drive(path):
	parts = path.split(":")
	return parts[0]


def get_real_ext(s):
	full_ext = s.split(".")
	if len(full_ext) > 1:
		real_ext = full_ext[1][:5].lower()
	else:
		real_ext = s[:5].lower()
	return real_ext

# declare generic functions
def depth_count(path):
	if path is None:
		return 0
	path = str(path)
	depth = path.count("\\")
	return depth


def string_length(s):
	length = len(str(s))
	return length

def entropy(string):
	# "Calculates the Shannon entropy of a string"
	# get probability of chars in string
	prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
	# calculate the entropy
	entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
	return entropy

def last_depth(path):
	if path is None:
		return ''
	parts = path.split("\\")
	return parts[-1]


#####################################################################################################################

def distinct_ext(column):
	# print ext_list
	xx = column.value_counts()
	xdict = xx.to_dict().items()
	threshold = len(column) * .005
	for k, v in xdict:
		if v > threshold:
			ext_list.append(k)

	dir = os.path.dirname(__file__)
	ext_file_name = os.path.join(dir, ext_file)
	with open(ext_file_name, 'wb') as fid:
		cPickle.dump(ext_list, fid)

def exist_matches(sub, artifact):
	if artifact is None or sub is None:
		return 0
	else:
		string = str(artifact).lower()
		if sub.lower() in string:
			return 1
		else:
			return 0
	return 0



def mimikatz(artifact):
	if artifact is None:
		return 'nomatch'
	string = str(artifact).lower()
	for i in mkatz_list:
		if i in string:
			return i
			break
	return 'nomatch'


def hasDir(artifact):
	if artifact is None:
		return 0
	string = str(artifact).lower()
	d = '\\'
	if d in string:
		return 1
	else:
		return 0



def format_size(size):
	try:
		if size is None or pd.isnull(size):
			return 0
		else:
			return long(size)
	except:
		return 0


def format_input(df):
	#  change all to str
	df['fullpath'] = str(df['fullpath'])
	df['textatlowestoffset'] = str(df['textatlowestoffset'])
	df['dataatlowestoffset'] = str(df['dataatlowestoffset'])
	df['username'] = str(df['username'])
	df['processpath'] = str(df['processpath'])
	df['pid'] = str(df['pid'])
	df['size'] = format_size(df['size'])
	# grab filename from path
	df['filename'] = last_depth(df['fullpath'])
	# grab extension from filename
	df['base_ext'] = get_real_ext(df['fullpath'])

	dir = os.path.dirname(__file__)
	process_file = os.path.join(dir, importance_process_file)
	all_important_processes = cPickle.load(open(process_file, 'rb'))

	for i in all_important_processes:
		colname = 'process=' + i
		df[colname] = exist_matches(i, df['process'])



	dir = os.path.dirname(__file__)
	ext_file_name = os.path.join(dir, ext_file)
	all_ext_list = cPickle.load(open(ext_file_name, 'rb'))

	for i in all_ext_list:
		colname = 'extension_contains_' + i
		df[colname] = exist_matches(i, df['base_ext'])

	for i in network_terms:
		colname = 'textbytes_contains_net_' + i
		df[colname] = exist_matches(i, df['textatlowestoffset'])

	for i in mag_bytes:
		colname = 'textbytes_contains_mag_' + i
		df[colname] = exist_matches(i, df['textatlowestoffset'])

	for i in txt_offset_IOCs:
		colname = 'textbytes_contains_offsetIOC_' + i
		df[colname] = exist_matches(i, df['textatlowestoffset'])

	for i in known_unames:
		colname = 'username=' + i
		df[colname] = exist_matches(i, df['username'])

	##########################################################################################################################
	# apply functions to filename and extension

	df['filenameLength'] = string_length(df['filename'])
	df['filenameEntropy'] = entropy(df['filename'])

	df['depth'] = depth_count(df['fullpath'])
	df['pathLength'] = string_length(df['fullpath'])
	df['pathEntropy'] = entropy(df['fullpath'])

	##########################################################################################################################
	# textatlowestoffset analytics:

	df['MZatLowestOffset'] = MZ(df['textatlowestoffset'])

	df['mimikatz'] = mimikatz(df['textatlowestoffset'])

	##########################################################################################################################
	df['hasDir'] = hasDir(df['textatlowestoffset'])

	##########################################################################################################################

	df['headerEntropy'] = entropy(df['textatlowestoffset'])
	df['offsetDataEntropy'] = entropy(df['dataatlowestoffset'])

	##########################################################################################################################

	df['systemPID'] = system_PID(df['pid'])
	# df['mcube_hits'] = md5_stat(df['md5'])

	# delete unused keys
	for key in delete_list:
		if key in df:
			df.pop(key)
	return df


# def md5_stat(md5):
# 	if md5 is None:
# 		return 0
# 	else:
# 		mcube_hits, mcube_total = client.get_mcube_statistics(md5)
# 		return mcube_hits

def format_dataframe(df):
	# Remove all columns that are extra
	for key in list(df.columns.values):
		if key not in file_keys and key != 'status':
			df.drop(key, axis=1)

	df['fullpath'] = df['fullpath'].astype(str)
	df['textatlowestoffset'] = df['textatlowestoffset'].astype(str)
	df['dataatlowestoffset'] = df['dataatlowestoffset'].astype(str)
	df['username'] = df['username'].astype(str)
	df['processpath'] = df['processpath'].astype(str)
	df['pid'] = df['pid'].astype(str)
	df['size'] = df['size'].apply(format_size)

	# grab filename from path
	df['filename'] = df['fullpath'].apply(last_depth)

	# grab extension from filename
	df['base_ext'] = df['fullpath'].apply(get_real_ext)

	distinct_ext(df['base_ext'])
	distinct_proc(df['process'])

	for i in important_processes:
		colname = 'process=' + i
		df[colname] = df['process'].apply(partial(exist_matches, i))

	for i in path_IOCs:
		colname = 'path_contains_' + i
		df[colname] = df['fullpath'].apply(partial(exist_matches, i))

	for i in filename_IOCs:
		colname = 'filename_contains_' + i
		df[colname] = df['filename'].apply(partial(exist_matches, i))

	for i in ext_list:
		colname = 'extension_contains_' + i
		df[colname] = df['base_ext'].apply(partial(exist_matches, i))

	for i in network_terms:
		colname = 'textbytes_contains_net_' + i
		df[colname] = df['textatlowestoffset'].apply(partial(exist_matches, i))

	for i in mag_bytes:
		colname = 'textbytes_contains_mag_' + i
		df[colname] = df['textatlowestoffset'].apply(partial(exist_matches, i))

	for i in txt_offset_IOCs:
		colname = 'textbytes_contains_offsetIOC_' + i
		df[colname] = df['textatlowestoffset'].apply(partial(exist_matches, i))

	for i in known_unames:
		colname = 'username=' + i
		df[colname] = df['username'].apply(partial(exist_matches, i))

	# get long tail dist of extensions
	##########################################################################################################################
	# apply functions to filename and extension

	df['filenameLength'] = df['filename'].apply(string_length)
	df['filenameEntropy'] = df['filename'].apply(entropy)

	df['depth'] = df['fullpath'].apply(depth_count)
	df['pathLength'] = df['fullpath'].apply(string_length)
	df['pathEntropy'] = df['fullpath'].apply(entropy)

	##########################################################################################################################
	# textatlowestoffset analytics:
	df['MZatLowestOffset'] = df['textatlowestoffset'].apply(MZ)
	df['mimikatz'] = df['textatlowestoffset'].apply(mimikatz)

	##########################################################################################################################

	df['hasDir'] = df['textatlowestoffset'].apply(hasDir)

	##########################################################################################################################

	df['headerEntropy'] = df['textatlowestoffset'].apply(entropy)
	df['offsetDataEntropy'] = df['dataatlowestoffset'].apply(entropy)

	##########################################################################################################################

	df['systemPID'] = df['pid'].apply(system_PID)
	# df['mcube_hits'] = df['md5'].apply(md5_stat)

	for i in delete_list:
		if i in list(df):
			df = df.drop(i, axis=1)

	df.fillna('NA', inplace=True)
	#
	# print "Columns = ", df.columns
	# print('Num data =', len(df))
	# print()
	# print('Feature - dtype')
	# print(df.dtypes)

	# print df.head()

	return df


df = create_features.format_input(row)


from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

model_cache = {}
model_client_predict = ModelDatabaseClient(user_type='predict')
model_client_train = ModelDatabaseClient(user_type='train')
alerts_db_client = AlertsDatabaseClient()


if sys.version[0] == '2':
	reload(sys)
	sys.setdefaultencoding("utf-8")


def set_current_model(model_name):
	model_type = model_client_train.set_current_model(model_name)
	if model_type and model_type in model_cache:
		model_cache[model_type] = None
		print 'Invalidating current value for model type, ', model_type
	return model_type

def get_sample_weights(df,y):
	ratio = 0.5
	weights = []

	if 'status' in df:
		res = column_vals(df, 'status')
		print "column values \n", res
		if 'threat' in res:
			print res['threat']
			threats = res['threat']
		else:
			threats = 0.0
		ratio = float(threats*1.0/len(y))
		# print "Ratio", ratio
		if ratio != 1.0:
			weight = 0.5 / ratio
			weights.append(weight)
			weight = 0.5 / (1 - ratio)
			weights.append(weight)
		else:
			raise ValueError('The dataset only has one type of status, cannot create model.')

	return weights

def print_distro(X, y):
	df_with_label = X
	df_with_label['status'] = y

	print "Total Rows=", df_with_label.shape
	df_threat = df_with_label[df_with_label['status'] == 'threat']
	df_fp = df_with_label[df_with_label['status'] == 'falsepositive']
	print "Threat", df_threat.shape
	print "FP", df_fp.shape

	feature_names = list(X.columns.values)

	for row in feature_names:
		if '_' in row:
			print'Column =', row

			if df_threat.shape[1] != 0:
				if row in df_threat:
					# print "Threat"
					res1 = column_vals(df_threat, row)

			if df_fp.shape[1] != 0:
				if row in df_fp:
					# print "False Positive"
					res2 = column_vals(df_fp, row)
			print "Threat = ", res1	, "\n FP = ", res2

def get_model(model_type):
	if model_type not in model_cache or model_cache[model_type] is None:
		model = model_client_predict.get_current_model(model_type=model_type)
		assert isinstance(model, ModelStorage)
		model_cache[model_type] = model
	else:
		model = model_cache[model_type]
	assert isinstance(model, ModelStorage)
	assert (model_type == model.model_type)
	return model


def make_prediction(df, model_type):
	try:
		model_info = get_model(model_type)
		loaded_vectorizer = model_info.vectorizer
		loaded_model = model_info.prediction_model
		### now predict label for the data ###

		if isinstance(df, dict):
			df_to_dict = df
		else:
			df_to_dict = df.to_dict(orient="records")

		to_predict = loaded_vectorizer.transform(df_to_dict)

		p = loaded_model.predict(to_predict)

		label = p[0]
		score = loaded_model.predict_proba(to_predict)
		score = max(score[0])

		pred = {
        		'label': label,
    			'score': float("{0:.2f}".format(100.00*score)),
				'model_name': model_info.model_name,
				'model_performance':  float("{0:.2f}".format(100.00*model_info.model_score)),
				'model_recall_score': float("{0:.2f}".format(100.00*model_info.recall_score))
				}
		# print score, label
		return pred

	except:
		print 'Error in making prediction'
		return None


def make_model(df, y, model_type, name=None, weight_samples=True):
	sample_weight = []
	weights = [1.0, 1.0]
	# Prints distribution of the features
	# print_distro(df, y)

	df, y = shuffle(df, y, random_state=0)

	if 'status' in df:
		print("Status present")
		if weight_samples is True:
			weights = get_sample_weights(df, y)
		df = df.drop('status', axis=1)

	assert isinstance(df, DataFrame)

	if isinstance(df, dict):
		df_to_dict = df
	else:
		df_to_dict = df.to_dict(orient="records")

	vec = DictVectorizer(sparse=False)
	# NOTE - tree-based learners should be able to handle categorical
	#        features without going through a preprocessing step,
	#        but that functionality is not currently in scikit-learn.
	#        Hopefully it's coming soon though with this:
	#        https://github.com/scikit-learn/scikit-learn/pull/4899
	vec.fit(df_to_dict)
	X = vec.transform(df_to_dict)

	print X.shape

	for label in y:
		if label == 'threat':
			sample_weight.append(weights[0])
		else:
			sample_weight.append(weights[1])

	rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced')
	rf_clf.fit(X, y, sample_weight=sample_weight)

	columns_names = vec.feature_names_
	# print len(columns_names)
	input_dataframe = DataFrame(data=X, columns=columns_names)
	# Remove all minor features
	threshold_for_features = 0.001
	imp = rf_clf.feature_importances_
	for index, value in enumerate(imp):
		if value < threshold_for_features:
			# print("dropping feature with little significance, = ", columns_names[index])
			input_dataframe = input_dataframe.drop(columns_names[index], axis=1)

	for feat in list(input_dataframe.columns.values):
		if feat.endswith("=") or feat.endswith("=NA"):
			print("dropping feature with no value = ", feat)
			input_dataframe = input_dataframe.drop(feat, axis=1)

	print input_dataframe.shape

	rf_clf_final = RandomForestClassifier(n_estimators=150, class_weight='balanced')
	df_to_dict = input_dataframe.to_dict(orient="records")
	vec.fit(df_to_dict)
	X_new = vec.transform(df_to_dict)
	# print vec.get_feature_names()
	# print len(vec.get_feature_names())
	cv_scores = cross_val_score(rf_clf_final, X_new, y, cv=10)

	print cv_scores
	print("Avg CV Score: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

	X_new, y = shuffle(X_new, y, random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.15, random_state=0)

	sample_weight = []
	for label in y_train:
		if label == 'threat':
			sample_weight.append(weights[0])
		else:
			sample_weight.append(weights[1])

	rf_clf_final.fit(X_train, y_train, sample_weight=sample_weight)

	y_pred = rf_clf_final.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print 'RF CM for Test Data Set:: \n', model_type
	print cm
	score = rf_clf_final.score(X_test, y_test)
	print "final test data accuracy score=",score
	lb = preprocessing.LabelBinarizer()
	lb.fit(y)
	yt = lb.transform(y_test)
	yp = lb.transform(y_pred)

	# print "CORRELATION MATRIX "
	# print input_dataframe.corr()
	model_recall_score = 0
	try:
		model_recall_score = recall_score(yt, yp, average='macro')
		auc_score = roc_auc_score(yt, yp)
		print("recall score", model_recall_score, " auc score ", auc_score)
	except:
		print "Error calculating recall score"


	feature_importances = sorted_feature_importances(rf_clf_final, vec, RandomForestClassifier)
	print feature_importances




def sorted_feature_importances(model, vectorizer, modelClass):
	assert isinstance(model, modelClass)
	assert isinstance(vectorizer, DictVectorizer)
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1]
	feature_names = np.asarray(vectorizer.get_feature_names())[indices]
	importances = importances[indices]
	feature_importance = DataFrame(
		{'feature': feature_names[0:20], 'importance': np.round(importances[0:20], 3)})
	feature_importance = feature_importance[abs(feature_importance['importance']) > 0.0]
	return feature_importance





def get_data_from_db():

	df = None
	y = None

	#####################################################################################################################
	############Read the data from reports db ###########################################################################
	engine = create_engine(app.DB_TRAIN_URL)

	query_string = """select
		status, signature, client_name, alert_uuid,
		COALESCE( data->'fileWriteEvent/fullPath','null') as fullpath,
		COALESCE( data->'fileWriteEvent/devicePath','null') as devicepath,
		COALESCE( data->'fileWriteEvent/filePath','null') as filepath,
		COALESCE( data->'fileWriteEvent/fileName','null') as filename,
		COALESCE( data->'fileWriteEvent/fileExtension','null') as fileextension,
		COALESCE( data->'fileWriteEvent/size','null') as size,
		COALESCE( data->'fileWriteEvent/md5','null') as md5,
		COALESCE( data->'fileWriteEvent/pid','null') as pid,
		COALESCE( data->'fileWriteEvent/process','null') as process,
		COALESCE( data->'fileWriteEvent/processPath','null') as processpath,
		COALESCE( data->'fileWriteEvent/dataAtLowestOffset','null') as dataatlowestoffset,
		COALESCE( data->'fileWriteEvent/textAtLowestOffset','null') as textatlowestoffset,
		COALESCE( data->'fileWriteEvent/username','null') as username,
		prediction_label
		from alerts where data_type = 'fileWriteEvent'
		and status in ('falsepositive','threat')
		and service = 'HX DTI'
		and occurred<'{0}' and occurred>='{1}'
		;"""

	end = datetime.today()
	delta = timedelta(hours=24 * 360)
	begin = end - delta
	query_string = query_string.format(end.strftime('%Y-%m-%d %H:%M:%S'), begin.strftime('%Y-%m-%d %H:%M:%S'))

	# print query_string

	with engine.connect() as con:
		rs = con.execute(query_string)
		results = rs.fetchall()
		# print "db "
		# print "result", results
		df = DataFrame(results, columns=rs.keys())
		df.drop_duplicates(subset='alert_uuid', keep='first', inplace=True)
		df.drop('alert_uuid', axis=1, inplace=True)
		df = df[df['signature'].str.contains('RANSOM') == False]
		df = df.drop('signature', axis=1)

		print "Without ransom = ",df.shape

		df = df[df['client_name'].str.contains('DevQA') == False]
		df = df.drop('client_name', axis=1)

		# double down on alerts that were wrongly predicted false positive before
		more_df = df[df['prediction_label'] != df['status'] ]
		print("prev wrong ones ", more_df.shape)
		if more_df.shape[0]>0:
			df = df.append(more_df)
		df = df.drop('prediction_label', axis=1)

		y = df['status']
		# df = df.drop('status', axis=1)

		return df, y


if __name__ == "__main__":
	df, y = get_data_from_db()


import pandas as pd
import json
from flask import make_response


def _json_response(obj):
	response = make_response(json.dumps(obj))
	response.headers['Content-Type'] = 'application/json'
	return response


def remove_nan(artifact):
	if pd.isnull(artifact):
		return 'NA'
	else:
		return artifact


def clean_database_issue(row):
	if type(row['data']) != unicode:
		return row
	data = json.loads(row['data'])
	# print row.keys()
	for item in data.keys():
		splits = item.split('/')
		if len(splits) > 1:
			key = item.split('/')[1].lower()
			row[key] = data[item]
	return row


def column_vals(df, col):
	res = df[col].value_counts()
	# print('Column        =', col)
	# print('# uniq values =', len(res))
	# print()
	# print(res)
	return res

