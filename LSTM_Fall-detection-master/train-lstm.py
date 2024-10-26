import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

number_video = 128  
sequence_length = 25
Duongdanlink = 'txtpre-processing'

actions = ['falltxt','lyingtxt','sittxt','standtxt','walktxt']
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    number=1
    for sequence in np.array(os.listdir(os.path.join(Duongdanlink, action))):
        if number <= 128:
            data = pd.read_csv(os.path.join(Duongdanlink, action, str(sequence)))
            dataset = data.iloc[:,1:].values
            sequences.append(dataset)
            number += 1
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
print(X.shape)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)

# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(25,132)))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(len(actions), activation='softmax'))

# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# model.save('bro.h5')
model = tf.keras.models.load_model('bro.h5')
res = model.predict(X_test)

print(actions[np.argmax(res[11])])
print(actions[np.argmax(y_test[11])])