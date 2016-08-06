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
        y = df.drop("Image", 1).as_matrix()
        y = (y - IMG_SIZE // 2) / (IMG_SIZE // 2)
        return shuffle(X, y)

    return X
