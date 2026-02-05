import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = []
y = []

files = [
    "fist", "open_palm", "thumb_up",
    "thumb_left", "thumb_right"
]

for f in files:
    X.append(np.load(f"{f}_X.npy"))
    y.append(np.load(f"{f}_y.npy"))

X = np.vstack(X)
y = np.hstack(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
np.save("scaler.npy", scaler.mean_)
