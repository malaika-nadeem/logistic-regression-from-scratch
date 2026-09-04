# 💀HALT! WHO GOES THERE?
#A tiny neural network standing guard at the login gate.*
#New device from an unfamiliar country? Not today.
# This project trains a small logistic regression model (built with Keras/TensorFlow) to look at a login attempt and decide whether it smells suspicious.

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from keras import Sequential
from keras.layers import Dense

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "HWGT.CSV")
from sklearn.model_selection import train_test_split

# feature scaling FIRST, on the raw column in df
df['login_hour_scaled'] = (df['login_hour'] - df['login_hour'].min()) / (df['login_hour'].max() - df['login_hour'].min())

# NOW select your features, using the already-scaled column
X = df[['login_hour_scaled', 'country_match', 'device_known']]
y = df['suspicious']

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# train-test split with stratification to maintain class distribution
model = Sequential([
    Dense(units=1, activation='sigmoid', input_shape=(3,))
])
# compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# train the model - now with class_weight to handle imbalance, and more epochs
model.fit(
    X_train, y_train,
    epochs=300,
    verbose=0,
    class_weight={0: 1, 1: 8}
)

# NEW: check real performance on the held-out test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

predictions = model.predict(X_train)
print("Predictions:", predictions[:10])  # just first 10 to keep output readable

def check_login(login_hour, country_match, device_known):
    # scale input
    scaled_hour = login_hour / 23

    user_inputs = np.array([[scaled_hour, country_match, device_known]])

    prediction = model.predict(user_inputs, verbose=0)
    probability = prediction[0][0]

    return probability
user = input("Enter login hour (0-23), country match (0/1), device known (0/1): ")

raw_inputs = [int(x) for x in user.split()]

probability = check_login(
    raw_inputs[0],
    raw_inputs[1],
    raw_inputs[2]
)

final_answer = 1 if probability > 0.5 else 0

print(f"Prediction: {final_answer}")
print(f"Confidence: {probability:.4f}")
#The Guard Stopped Guessing
#Loss flatlines by epoch 50 — decision made, gate secured.
history = model.fit(X_train, y_train, epochs=500)

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.title('Loss over Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()