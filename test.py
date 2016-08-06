from collections import OrderedDict
from keras.models import model_from_json
from utils import process_data
from utils import TEST_PATH, MODEL_PATH, WEIGHTS_PATH, LOOKUP_PATH, SUBMISSION_PATH, IMG_SIZE, BATCH_SIZE, SPECIALIST_SETTINGS
import numpy as np
import pandas as pd


def parse_lookup_table(fname):
    lookup = pd.read_csv(fname)
    feature_index = {}
    for i in range(30):
        feature_index[lookup["FeatureName"][i]] = i
    return lookup, feature_index 


def load_model():
    model = model_from_json(open(MODEL_PATH).read())
    model.load_weights(WEIGHTS_PATH)
    model.compile(loss="mse", optimizer="sgd")
    

def load_specialists():
    specialists = OrderedDict()
    for setting in SPECIALIST_SETTINGS:
        cols = setting["columns"]
        model_path = "data/model_{}.json".format(cols[0])
        model = model_from_json(open(model_path).read())
        weights_path = "data/weights_{}.h5".format(cols[0])
        model.load_weights(weights_path)
        specialists[cols] = model
    return specialists




if __name__ == "__main__":
    lookup, feature_index = parse_lookup_table(LOOKUP_PATH)
    X = process_data(TEST_PATH, mode="TEST")
    specialists = load_specialists()
    predictions = {}

    for cols, model in specialists.items():
        spec_predictions = model.predict(X, batch_size=BATCH_SIZE)
        spec_predictions *= IMG_SIZE // 2
        spec_predictions += IMG_SIZE // 2
        for i, col in enumerate(cols):
            predictions[col] = spec_predictions[:,i]

    submission_values = []
    for i in range(len(lookup)):
        img_id = lookup["ImageId"][i] - 1
        feature = lookup["FeatureName"][i]
        submission_values.append(predictions[feature][img_id])

    submission = pd.DataFrame({"RowId": np.arange(1, len(lookup) + 1), "Location": submission_values})
    submission = submission[["RowId", "Location"]]
    submission.to_csv(SUBMISSION_PATH, index=False)
