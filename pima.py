import tensorflow as tf
import  pandas as pd
import numpy as np

data = pd.read_csv('diabetes.csv')


x_train = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y_train = data['Outcome']

x_train = tf.keras.utils.normalize(x_train,axis=1)
#y_train = tf.keras.utils.normalize(y_train,axis=1)


x_train = np.array(x_train)
y_train = np.array(y_train)

"""print(x_train)
print(y_train)"""
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(10,input_dim=8,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(400,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(300,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1000)

