import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_and_save(version="v1", path="model_v1.joblib", random_state=42):
    # Seeded so the artifact is byte-reproducible; an unseeded artifact makes
    # the deployment lab impossible to diff across runs.
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((100, 4))
    y = (X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(0, 0.5, 100) > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    joblib.dump(model, path)
    return model

if __name__ == "__main__":
    train_and_save()
