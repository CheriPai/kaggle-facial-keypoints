from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
import numpy as np
import pandas as pd


train_path = "data/training.csv"


def process_data(fname):
    df = pd.read_csv("data/training.csv")
    images = df.as_matrix(columns=["Image"])
    X_train = np.array([np.array(row[0].split(" ")) for row in images])
    X_train = X_train.astype(np.int).reshape(X_train.shape[0], 1, 96, 96)
    y_train = df.drop("Image", 1).fillna(df.median()).as_matrix()
    return X_train, y_train


def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 96, 96)))
    model.add(Activation("relu"))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode="valid"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.50))

    model.add(Dense(30))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    return model


if __name__ == "__main__":
    X_train, y_train = process_data(train_path)
    model = build_model()
    model.fit(X_train, y_train, nb_epoch=10, batch_size=128)
    mse = model.evaluate(X_train, y_train, batch_size=128)
    print("RMSE: ", np.sqrt(mse))
