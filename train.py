from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
from utils import process_data
from utils import TRAIN_PATH, MODEL_PATH, WEIGHTS_PATH, BATCH_SIZE, IMG_SIZE, VAL_PROP
import numpy as np
import pandas as pd


def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, IMG_SIZE, IMG_SIZE)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Convolution2D(64, 2, 2, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 2, 2, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.50))

    model.add(Dense(1024))
    model.add(Activation("relu"))

    model.add(Dense(30))

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    return model


if __name__ == "__main__":
    X, y = process_data(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)
    model = build_model()
    model.fit(X_train, y_train, nb_epoch=1000, batch_size=BATCH_SIZE, verbose=1)

    print("Saving model to ", MODEL_PATH)
    print("Saving weights to ", WEIGHTS_PATH)
    open(MODEL_PATH, 'w').write(model.to_json())
    model.save_weights(WEIGHTS_PATH)

    mse = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
    print("RMSE: ", np.sqrt(mse)*IMG_SIZE)
