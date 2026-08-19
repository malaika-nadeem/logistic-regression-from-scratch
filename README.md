# Logistic Regression from Scratch — Student Pass/Fail Prediction

A logistic regression model built entirely from scratch in NumPy — no `sklearn`,
no black-box `.fit()`. Every piece (scaling, sigmoid, cost, gradient, gradient
descent) is implemented and derived by hand.

## What it does

Predicts whether a student passes or fails based on two features:
- Hours studied
- Attendance (%)

## Dataset

A small, synthetic dataset of 16 students (12 train / 4 test). The data is
**linearly separable** — every student below 4 study hours failed, every
student at or above 4 hours passed, with no overlap. This makes 100% accuracy
achievable and expected; it's not evidence of a "strong" model, just an easy
dataset. Worth knowing before assuming this generalizes to messier real-world
data.

## Pipeline

1. **Split** data into train/test sets
2. **Scale** features with min-max normalization — `(x - min) / (max - min)` —
   using min/max computed from training data only, then reused to scale test
   data (prevents data leakage)
3. **Forward pass** — `z = w·x + b`, then `sigmoid(z)` for probabilities
4. **Cost function** — binary cross-entropy / log-loss
5. **Gradient descent** — iteratively updates `w` and `b` to minimize cost
6. **Evaluate** — accuracy on both train and held-out test data

## Results

- Training accuracy: 100%
- Test accuracy: 100%
- Cost converges from ~0.66 to ~0.06 over 5000 iterations (see cost curve
  in notebook)

## What I learned

- The difference between mean normalization and true min-max scaling, and why
  mislabeling one as the other is an easy silent bug
- Why a badly chosen bias term can make a model predict the same class for
  every input, regardless of the actual data
- Why min/max used for scaling must always come from training data — using
  test data's own range leaks information the model shouldn't have
- On linearly separable data, gradient descent pushes weights toward large
  magnitudes rather than converging to small, "clean" numbers — this is
  expected behavior, not a bug

## Tech

Python, NumPy, Pandas, Matplotlib

## Files

- `student_prediction.ipynb` — full pipeline, code + explanations
- `student_data.csv` — dataset