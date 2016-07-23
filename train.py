from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd


train_path = "data/training.csv"
IMG_SIZE = 96
VAL_PROP = 0.1


def process_data(fname):
    df = pd.read_csv(fname)
    # Should we impute or drop?
    # df = df.dropna()
    imgs = df.as_matrix(columns=["Image"])
    X = np.array([np.array(row[0].split(" ")) for row in imgs])
    X = X.astype(np.float).reshape(X.shape[0], 1, IMG_SIZE, IMG_SIZE)
    X /= 255
    y = df.drop("Image", 1).fillna(df.mean()).as_matrix()
    y /= IMG_SIZE
    return X, y


def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, IMG_SIZE, IMG_SIZE)))
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

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    return model


if __name__ == "__main__":
    X, y = process_data(train_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP, random_state=123)
    model = build_model()
    model.fit(X_train, y_train, nb_epoch=4, batch_size=160)
    mse = model.evaluate(X_val, y_val, batch_size=160)
    print("RMSE: ", np.sqrt(mse))
