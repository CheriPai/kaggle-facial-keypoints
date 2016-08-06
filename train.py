from collections import OrderedDict
from keras.callbacks import EarlyStopping
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


SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_idxs=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_idxs=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_idxs=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_idxs=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_idxs=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_idxs=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]


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


def train_model(pretrain):
    X, y = process_data(TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)

    if pretrain:
        model = model_from_json(open(MODEL_PATH).read())
        model.load_weights(WEIGHTS_PATH)
    else:
        model = build_model()

    flipgen = FlippedImageDataGenerator()
    sgd = SGD(lr=0.08, decay=1e-4, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd)
    early_stop = EarlyStopping(monitor="val_loss", patience=100, mode="min")
    model.fit_generator(flipgen.flow(X_train, y_train),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=5000,
                        validation_data=(X_val, y_val),
                        callbacks=[early_stop])

    print("Saving model to ", MODEL_PATH)
    print("Saving weights to ", WEIGHTS_PATH)
    open(MODEL_PATH, 'w').write(model.to_json())
    model.save_weights(WEIGHTS_PATH, overwrite=True)

    mse = model.evaluate(X_val, y_val, batch_size=BATCH_SIZE)
    print("MSE: ", mse)
    print("RMSE: ", np.sqrt(mse)*IMG_SIZE)


def train_specialists(pretrain):
    specialists = OrderedDict()
    for setting in SPECIALIST_SETTINGS:
        cols = setting["columns"]
        X, y = process_data(TRAIN_PATH, cols)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_PROP)

        if pretrain:
            model = model_from_json(open(MODEL_PATH).read())
            model.load_weights(WEIGHTS_PATH)
        else:
            model = build_model()

        model.layers.pop()
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []
        model.add(Dense(len(cols), name="dense_3"))

        flipgen = FlippedImageDataGenerator()
        flipgen.flip_idxs = setting["flip_idxs"]
        sgd = SGD(lr=0.08, decay=1e-4, momentum=0.9, nesterov=True)
        model.compile(loss="mse", optimizer=sgd)
        early_stop = EarlyStopping(monitor="val_loss", patience=100, mode="min")
        print("Training {}...".format(cols[0]))
        model.fit_generator(flipgen.flow(X_train, y_train),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=1000,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stop])

        weights_path = "data/weights_{}.h5".format(cols[0])
        print("Saving weights to ", weights_path)
        model.save_weights(weights_path, overwrite=True)




# RMSE: 2.597
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true')
    args = parser.parse_args()

    # train_model(args.p)
    train_specialists(args.p)

