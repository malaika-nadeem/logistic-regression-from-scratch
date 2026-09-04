# 💀HALT! WHO GOES THERE?

*A tiny neural network standing guard at the login gate.*

New device from an unfamiliar country? Not today. This project trains a small logistic regression model (built with Keras/TensorFlow) to look at a login attempt and decide whether it smells suspicious.

---

## What it does

The model checks three things about every login:

- **When** — what hour the login happened
- **Where** — whether the login is coming from the user's usual country
- **What** — whether the device is recognized or brand new

...and outputs a probability that the login is suspicious.

## The rule it learned

```
suspicious = 1   if device is NEW  AND  country is UNUSUAL
suspicious = 0   otherwise
```

Only the **combination** of both risk factors together counts as suspicious — a new phone from your home country is fine, and your old laptop showing up abroad is fine too. It's the two together that raise the flag.

## How it's built

- **Data:** 2000 synthetic login records generated from the rule above, with 5% label noise added to keep it realistic
- **Preprocessing:** `login_hour` manually min-max scaled to a 0–1 range
- **Split:** 70/30 train/test, stratified to preserve the ~11% suspicious class ratio
- **Model:** a single-layer logistic regression (`Dense(1, activation='sigmoid')`) — no hidden layers needed, since this rule is linearly separable
- **Training:** Adam optimizer, binary crossentropy loss, class weighting to counter the class imbalance

## Try it yourself

Run the notebook, and at the prompt enter three values — hour (0–23), country match (0/1), device known (0/1) — and the guard will tell you whether it's letting you through.

---

*Part of the [Lumera](https://github.com/malaika-nadeem/LUMERA) project collection.*