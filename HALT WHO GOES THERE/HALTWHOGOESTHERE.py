# 💀HALT! WHO GOES THERE?
# A tiny neural network standing guard at the login gate.
# New device from an unfamiliar country? Not today.
# This project trains a small softmax classifier (built with Keras/TensorFlow) to look at a
# login attempt and decide whether it's a legitimate user, an impersonator, or a bot.

import time

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pathlib import Path

data_path = Path(__file__).parent / "halt_who_goes_there_logins.csv"
df = pd.read_csv(data_path)
print(df.shape)
print(df.tail(10))
start = time.time()
df = pd.read_csv(data_path)
print(f"Loading data took {time.time() - start:.2f} seconds")

# NEW: cyclical hour encoding instead of linear min-max scaling.
# Linear scaling made hour 23 and hour 0 look maximally far apart (1.0 vs 0.0),
# even though they're actually right next to each other on a 24-hour clock.
# Sine/cosine wraps the hour around a circle so midnight neighbors stay close together.
df['hour_sin'] = np.sin(2 * np.pi * df['login_hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['login_hour'] / 24)

# NOW select your features — hour_sin/hour_cos replace login_hour_scaled (4 features -> 5)
X = df[['hour_sin', 'hour_cos', 'country_match', 'device_known', 'email_matches_account']]

# labels are text (legitimate user / impersonator / bot), so encode them to integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])
print("Class mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# train-test split with stratification to maintain class distribution

# hidden layer (16 units) + softmax output layer. input_shape is now 5, not 4.
model = Sequential([
    Dense(units=16, activation='relu', input_shape=(5,)),
    Dense(units=3, activation='softmax')
])
# compile the model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
# train the model
model.fit(
    X_train, y_train,
    epochs=100,
    verbose=1,
)

# check real performance on the held-out test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

predictions = model.predict(X_train)
print("Predictions (first 10 rows, one probability per class):")
print(predictions[:10])


def check_login(login_hour, country_match, device_known, email_matches_account):
    # NEW: cyclical encoding for the single input hour, matching training preprocessing
    hour_sin = np.sin(2 * np.pi * login_hour / 24)
    hour_cos = np.cos(2 * np.pi * login_hour / 24)

    user_inputs = np.array([[hour_sin, hour_cos, country_match, device_known, email_matches_account]])

    prediction = model.predict(user_inputs, verbose=0)
    class_probabilities = prediction[0]

    predicted_class_index = np.argmax(class_probabilities)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    confidence = class_probabilities[predicted_class_index]

    return predicted_label, confidence, class_probabilities


user = input("Enter login hour (0-23), country match (0/1), device known (0/1), email matches account (0/1): ")

raw_inputs = [int(x) for x in user.split()]

predicted_label, confidence, class_probabilities = check_login(
    raw_inputs[0],
    raw_inputs[1],
    raw_inputs[2],
    raw_inputs[3]
)

print(f"Prediction: {predicted_label}")
print(f"Confidence: {confidence:.4f}")
print(f"All class probabilities: {dict(zip(label_encoder.classes_, class_probabilities))}")

if predicted_label == "bot":
    print("Stage 1: This login attempt looks automated — likely a bot.")
elif predicted_label == "impersonator":
    print("Stage 2: This login attempt looks like a possible impersonator.")
else:
    print("This login attempt looks like the legitimate user.")

# The Guard Stopped Guessing
# Loss flatlines by epoch 50 — decision made, gate secured.
history = model.fit(X_train, y_train, epochs=1000)

plt.plot(history.history['loss'])
plt.title('Loss over Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()