import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = [], []
files = ["fist", "open_palm", "thumb_up", "thumb_left", "thumb_right"]

for f in files:
    try:
        X.append(np.load(f"dataset/{f}_X.npy"))
        y.append(np.load(f"dataset/{f}_y.npy"))
    except FileNotFoundError:
        print(f"Warning: {f} files not found. Skipping.")

X = np.vstack(X)
y = np.hstack(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)


with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)
print("Data prepared and scaler saved.")