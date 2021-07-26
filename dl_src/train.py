import numpy as np 
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

file_train_x = open("dl_src/data/train_x.pickle", "rb")
train_x = pickle.load(file_train_x)

file_train_y = open("dl_src/data/train_y.pickle", "rb")
train_y = pickle.load(file_train_y)

file_test_x = open("dl_src/data/test_x.pickle", "rb")
test_x = pickle.load(file_test_x)

file_test_y = open("dl_src/data/test_y.pickle", "rb")
test_y = pickle.load(file_test_y)

input_dim = train_x.shape[1]

print("Training with model 1 in stack...")
in_data_1 = tf.keras.Input(shape=(input_dim,))
encoded_1 = tf.keras.layers.Dense(14, activation='relu')(in_data_1)
encoded_1 = tf.keras.layers.Dense(28, activation='relu')(encoded_1)
encoded_1 = tf.keras.layers.Dense(28, activation='relu')(encoded_1)
out_layer_1 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded_1)

model_1 = tf.keras.Model(inputs=in_data_1, outputs=out_layer_1)

model_1.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_1.summary()
model_1.fit(train_x, train_x, epochs=200, batch_size=64, shuffle=True)

print("Saving model 1 to folder...")
model_1.save('dl_src/models/model_1.h5')
out_1 = model_1.predict(train_x)


print("Training with model 2 in stack...")
in_data_2 = tf.keras.Input(shape=(input_dim,))
encoded_2 = tf.keras.layers.Dense(14, activation='relu')(in_data_2)
encoded_2 = tf.keras.layers.Dense(28, activation='relu')(encoded_2)
encoded_2 = tf.keras.layers.Dense(28, activation='relu')(encoded_2)
out_layer_2 = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded_2)

model_2 = tf.keras.Model(inputs=in_data_2, outputs=out_layer_2)

model_2.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_2.summary()
model_2.fit(out_1, train_x, epochs=200, batch_size=64, shuffle=True)

print("Saving model 2 to folder...")
model_2.save('dl_src/models/model_2.h5')
out_2 = model_2.predict(out_1)

print("Training with Random Forest...")
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}

rf_model = RandomForestClassifier(**parameters)
rf_model.fit(out_2, train_y)

print("Saving Random Forest model...")
pickle.dump(rf_model, open('dl_src/models/rf_model.sav', 'wb'))

print("Done!")
