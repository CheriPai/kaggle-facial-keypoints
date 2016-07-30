from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from utils import process_data
from utils import TRAIN_PATH, MODEL_PATH, WEIGHTS_PATH, BATCH_SIZE, IMG_SIZE, VAL_PROP
import argparse
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
    return model


class FlippedImageDataGenerator(ImageDataGenerator):
    flip_idxs = [ (0, 2), (1, 3), (4, 8), (5, 9),
                  (6, 10), (7, 11), (12, 16), (13, 17),
                  (14, 18), (15, 19), (22, 24), (23, 25) 
                ] 

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        idxs = np.random.choice(batch_size, batch_size / 2, replace=False)
        # Flip image horizontally
        X_batch[idxs] = X_batch[idxs, :, :, ::-1]

        if y_batch is not None:
            y_batch[idxs, ::2] = y_batch[idxs, ::2] * -1

            for a, b in self.flip_idxs:
                y_batch[idxs, a], y_batch[idxs, b] = (y_batch[idxs, b] , y_batch[idxs, a])

        return X_batch, y_batch



# lr 0.10, decay 1e-4 3.407 data aug
# lr 0.09, decay 1e-4 3.306 data aug
# lr 0.08, decay 1e-4 3.022 data aug
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true')
    args = parser.parse_args()

    X, y = process_data(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)

    if args.p:
        model = model_from_json(open(MODEL_PATH).read())
        model.load_weights(WEIGHTS_PATH)
    else:
        model = build_model()

    flipgen = FlippedImageDataGenerator()
    sgd = SGD(lr=0.07, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    model.fit_generator(flipgen.flow(X_train, y_train),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=1000,
                        verbose=1)

    print("Saving model to ", MODEL_PATH)
    print("Saving weights to ", WEIGHTS_PATH)
    open(MODEL_PATH, 'w').write(model.to_json())
    model.save_weights(WEIGHTS_PATH)

    mse = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
    print("MSE: ", mse)
    print("RMSE: ", np.sqrt(mse)*IMG_SIZE)
