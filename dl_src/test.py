import numpy as np 
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def evaluate(predictions, labels):
  TP = 0; FP = 0; TN = 0; FN = 0;

  for pair in zip(predictions, labels):
    if pair[0] != 0 and pair[1] != 0:
      TP += 1
    if pair[0] != 0 and pair[1] == 0:
      FP += 1
    if pair[0] == 0 and pair[1] == 0:
      TN += 1
    if pair[0] == 0 and pair[1] != 0:
      FN += 1

  accuracy = (TP + TN)/(TP + TN + FP + FN)
  precision = TP/(TP + FP)
  recall = TP/(TP + FN)
  false_adam = FP/(FP + TN)
  f1 = 2 * precision * recall / (precision + recall)

  return accuracy, precision, recall, false_adam, f1

if __name__ == '__main__':
  print('Loading models from files...', end='')
  model_1 = tf.keras.models.load_model('dl_src/models/model_1.h5')
  model_2 = tf.keras.models.load_model('dl_src/models/model_2.h5')
  rf_model = pickle.load(open('dl_src/models/rf_model.sav', 'rb'))
  print('Done')

  print('Loading test data from files...', end='')
  test_x = pickle.load(open('dl_src/data/test_x.pickle', 'rb'))
  test_y = pickle.load(open('dl_src/data/test_y.pickle', 'rb'))
  print('Done')

  print('Predicting...', end='')
  predictions = model_1.predict(test_x)
  predictions = model_2.predict(predictions)
  predictions = rf_model.predict(predictions)
  print('Done')

  accuracy, precision, recall, false_adam, f1 = evaluate(predictions, test_y)
  print("accuracy = ", accuracy)
  print("precision = ", precision)
  print("recall = ", recall)
  print("false_adam = ", false_adam)
  print("f1 = ", f1)

  print("Done!")
