from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
from itertools import chain
import pandas as pd
from pandas import DataFrame
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix,recall_score
from sklearn.metrics import matthews_corrcoef



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# fix random seed for reproducibility
numpy.random.seed(7)

def make_prediction(df, model, scaler):
	try:
		loaded_model = model
		print('inside make prediction')
		### now predict label for the data ###
		X = df.values
		X = scaler.transform(X)
		p = loaded_model.predict_proba(X)
		return p
	except:
		print('Error in making prediction')
		return None

def make_model(df):

	key = 'Criminal'
	# df[key] = df[key].astype(str)

	y = df['Criminal']
	df = df.drop('Criminal', axis = 1)
	#  Sto if loss is not changing

	earlystop = EarlyStopping(monitor='val_loss'
							  ,
							  mode='auto'
							, patience=5
							  )

	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
								  patience=2, min_lr=0.001)

	callbacks_list = [
		earlystop
		,
		reduce_lr
	]

	# create model
	model = Sequential()
	model.add(Dense(128, input_dim=df.shape[1], activation='relu'))
	model.add(Dropout(0.5))
	# model.add(Dense(100, activation='relu'))
	# model.add(Dropout(0.2))
	# model.add(Dense(64, activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	opt = optimizers.SGD()
	model.compile(loss='binary_crossentropy',
				  optimizer=opt,
				  metrics=['accuracy'])


	X = df.values

	scaler = StandardScaler().fit(X)
	X = scaler.transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
	model.fit(X_train, y_train,
			  epochs=500,
			  batch_size=256,
			  callbacks=callbacks_list,
			  validation_split=0.2,
			  verbose=2,
			  shuffle=True
			  )

	scores = model.evaluate(X_test, y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

	y_pred = model.predict_classes(X_test)

	print " \n y _ pred ", model.predict_proba(X_test)[:15]

	y_p = list(chain.from_iterable(y_pred))

	print "\nPred ", y_p[:15]

	print "\nTest ", y_test.values[:15]


	print " CM = \n", confusion_matrix(y_test.values, y_pred)

	try:
		model_recall_score = recall_score(y_test.values, y_p, average='macro')
		auc_score = roc_auc_score(y_test.values, y_pred)
		mcc = matthews_corrcoef(y_test.values, y_pred)
		print("recall score", model_recall_score, " auc score ", auc_score, " MCC SCore ", mcc)
	except Exception as e:
		print("Error calculating recall score", e)


	return model, scaler

#  _ big_4.csv is gold, keep it. do not replace/ rewrite
def get_data(name):
	file = name + '_encoded.csv'
	df = pd.read_csv(file)
	print(name + '  file read ', df.shape)
	df.fillna('NA', inplace=True)
	# print('To delete = ',to_delete)
	return df


if __name__ == "__main__":

	df = get_data('train')
	# df['Criminal'] = df['Criminal'].astype(str)

	model, scaler= make_model(df)
	print("train df shape ", df.shape)

	df_t = get_data('test')
	label = make_prediction(df_t, model, scaler)
	print("df test shape ", df_t.shape)

	scores = list(chain.from_iterable(label.tolist()))

	index = range(0, len(scores))
	print(len(index))

	id_field = 'PERID'

	df = DataFrame({id_field: index, 'Criminal': scores})
	print(df.head())
	df.to_csv(path_or_buf='submission_keras_sgd_small-combined_proba_27_noAge_1000'
						  '.csv', index=False)