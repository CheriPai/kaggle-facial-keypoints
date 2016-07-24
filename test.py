from keras.models import model_from_json
from utils import process_data
from utils import TEST_PATH, MODEL_PATH, WEIGHTS_PATH, LOOKUP_PATH, SUBMISSION_PATH, BATCH_SIZE
import numpy as np
import pandas as pd


def parse_lookup_table(fname):
    lookup = pd.read_csv(fname)
    feature_index = {}
    for i in range(30):
        feature_index[lookup["FeatureName"][i]] = i
    return lookup, feature_index 


if __name__ == "__main__":
    X = process_data(TEST_PATH, mode="TEST")
    lookup, feature_index = parse_lookup_table(LOOKUP_PATH)

    model = model_from_json(open(MODEL_PATH).read())
    model.load_weights(WEIGHTS_PATH)
    model.compile(loss="mse", optimizer="sgd")

    predictions = model.predict(X, batch_size=BATCH_SIZE)

    submission_values = []
    for i in range(len(lookup)):
        img_id = lookup["ImageId"][i] - 1
        index = feature_index[lookup["FeatureName"][i]]
        submission_values.append(predictions[img_id][index])

    submission = pd.DataFrame({"RowId": np.arange(1, len(lookup) + 1), "Location": submission_values})
    submission = submission[["RowId", "Location"]]
    submission.to_csv(SUBMISSION_PATH, index=False)
