import tensorflow as tf
from keras import layers, models, callbacks
import numpy as np

X_train, X_test = np.load("X_train.npy"), np.load("X_test.npy")
y_train, y_test = np.load("y_train.npy"), np.load("y_test.npy")

model = models.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop])

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")
model.save("gesture_model.h5")
