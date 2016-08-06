from sklearn.utils import shuffle
import numpy as np
import pandas as pd


TRAIN_PATH = "data/training.csv"
TEST_PATH = "data/test.csv"
MODEL_PATH = "data/model.json"
WEIGHTS_PATH = "data/weights.h5"
LOOKUP_PATH = "data/IdLookupTable.csv"
SUBMISSION_PATH = "data/submission.csv"
BATCH_SIZE = 128
IMG_SIZE = 96
VAL_PROP = 0.1


def process_data(fname, cols=None, mode="TRAIN"):
    """ Reads csv file and returns a numpy array of the image
        and corresponding values for the keypoints
    """
    df = pd.read_csv(fname)
    if cols:
        df = df[list(cols) + ["Image"]]
    df = df.dropna()
    imgs = df.as_matrix(columns=["Image"])
    X = np.array([np.array(row[0].split(" ")) for row in imgs])
    X = X.astype(np.float).reshape(X.shape[0], 1, IMG_SIZE, IMG_SIZE)
    X /= 255

    if mode == "TRAIN":
        # Should we impute or drop?
        y = df.drop("Image", 1).fillna(df.mean()).as_matrix()
        y = df.drop("Image", 1).as_matrix()
        y = (y - IMG_SIZE // 2) / (IMG_SIZE // 2)
        return shuffle(X, y)

    return X
