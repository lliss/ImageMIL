import argparse
import pickle

import numpy as np
import sklearn.metrics
import sklearn.model_selection
from linear_classifier import LinearClassifier
from sil import SIL


def main(features_file, model_file):
    features = [np.load(features_file)]
    features = np.concatenate(features, axis=0)
    if len(features.shape) == 1:
        features = features.reshape((1, len(features)))

    # compute mean if needed
    if len(features.shape) > 1:
        features = features.mean(axis=0)


    test_set = [features]

    with open(model_file, 'rb') as input_file:
        model = pickle.load(input_file)

    p_predict = model.predict(test_set)
    y_predict = np.argmax(p_predict, axis=1)
    print(p_predict)
    print(y_predict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser( description='Compute CNN features.' )
    parser.add_argument('features', nargs=1, help='a features file to load' )
    parser.add_argument('model', nargs=1, help='a model file to load' )
    args = parser.parse_args()
    main(args.features[0], args.model[0])
