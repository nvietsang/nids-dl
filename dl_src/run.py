import numpy as np 
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from preprocessing import pre_data, get_label_dict

if __name__ == '__main__':
	print('Loading models from files...', end='')
	model_1 = tf.keras.models.load_model('dl_src/models/model_1.h5')
	model_2 = tf.keras.models.load_model('dl_src/models/model_2.h5')
	rf_model = pickle.load(open('dl_src/models/rf_model.sav', 'rb'))
	print('Done')

	sample = pd.read_csv('dl_src/data/sample.txt', header=None,delim_whitespace=True)
	# print(sample)
	print(sample.dtypes)
	trash = pd.read_csv('dl_src/data/trash.txt', header=None)
	print("Raw train data has a shape of ", sample.shape)

	in_data = pre_data(sample)
	# for i in range(len(in_data)):    
	predictions = model_1.predict(np.array(in_data))
	predictions = model_2.predict(predictions)
	predictions = rf_model.predict(predictions)

	# print(predictions)
	label_dict = get_label_dict()
	for p in predictions:
		predicted_label = 'normal'
		for key, value in label_dict.items():
			if value == p:
				predicted_label = key
				break

		print('Predicted label is data', predicted_label)
  
