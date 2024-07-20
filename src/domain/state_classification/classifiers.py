# Ignore unused imports, as they will be used in the evaluation of the pipeline.

import os
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import (Binarizer, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, StandardScaler,
                                   QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, SplineTransformer, KBinsDiscretizer)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, accuracy_score


class CustomizedClassifier:
    def __init__(self, dataset_path: str, customization_path: str,
                 output_path: str, exact_index_path: str, train_set_size: int) -> None:
        # Init variables
        self.total_accuracy, self.precision_opened, self.precision_closed = None, None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.pipeline = None
        self.train_set_size = train_set_size
        self.exact_index_path = exact_index_path
        self.output_path = output_path

        self.get_dataset(dataset_path)
        self.get_pipeline(customization_path)
        self.predictions = self.get_prediction()
        self.results = self.get_results(self.predictions)
        self.save_results()

    def get_dataset(self, dataset_path: str) -> None:
        """
        Load the dataset into the classifier, obtaining the training and test sets.
        :param dataset_path: path of the dataset obtained with the DataLoader.
        """
        d = np.loadtxt(dataset_path, delimiter=",")
        x = d[:, 0:2]
        y = d[:, 2]

        # Take the middle windows for testing, and the rest for training the classifier
        division_index_left = int(len(x) * self.train_set_size / 200)
        test_size = len(x) - division_index_left*2
        division_index_right = division_index_left + test_size
        self.X_train, self.X_test = (
            np.concatenate(
                (x[:division_index_left],
                 x[division_index_right:])),
            x[division_index_left:division_index_right]
        )
        self.y_train, self.y_test = (
            np.concatenate(
                (y[:division_index_left],
                 y[division_index_right:])),
            y[division_index_left:division_index_right]
        )

    def get_pipeline(self, customization_path: str) -> None:
        """
        Load the pipeline with preprocessor and optimized classifier.
        :param customization_path: path where the csv file with the optimization is stored.
        """
        command = np.loadtxt(customization_path, dtype=str, delimiter=";")
        preprocessor = eval(command[1])
        classifier = eval(command[0])
        pipe_tuple = preprocessor + (classifier,)
        self.pipeline = make_pipeline(*pipe_tuple)

    def get_prediction(self) -> np.ndarray:
        """
        Trains the classifier with the stored pipeline and returns the array with the predicted classes.
        :return: Array with the predicted classes for the test set.
        """
        # Train the classifier
        self.pipeline.fit(self.X_train, self.y_train)

        # Use classifier on the test set
        y_pred = self.pipeline.predict(self.X_test)

        return y_pred

    def get_results(self, predictions: np.ndarray) -> dict:
        """
        Get a dict with the computed accuracy and precisions for the types of time windows.
        :param predictions: predicted classes in the test set.
        :return: dict with the fields: precision_closed, precision_opened, accuracy, real_tags_in_test, predicted_tags_in_test
        """
        exact_indexes = np.loadtxt(self.exact_index_path, delimiter=",")
        exact_predictions = []
        exact_test = []

        for i in range(len(predictions)):
            if i in exact_indexes:
                exact_predictions.append(predictions[i])
                exact_test.append(self.y_test[i])

        self.precision_closed = precision_score(exact_test, exact_predictions, pos_label=0)
        self.precision_opened = precision_score(exact_test, exact_predictions, pos_label=1)
        self.total_accuracy = accuracy_score(self.y_test, predictions)
        return {
            "precision_closed": self.precision_closed,
            "precision_opened": self.precision_opened,
            "accuracy": self.total_accuracy,
            "real_tags_in_test": self.y_test,
            "predicted_tags_in_test": predictions
        }

    def save_results(self):
        """
        Saves the results in the configured output path.
        """
        # Avoid errors with nonexistent dirs
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))

        with open(self.output_path, 'w') as f:
            f.write("precision_closed;precision_opened;accuracy;real_tags_in_test;predicted_tags_in_test\n")
            f.write(str(self.precision_closed) + ";" + str(self.precision_opened) + ";" + str(self.total_accuracy) +
                    ";" + str(self.y_test) + ";" + str(self.predictions) + ";" + "\n")
            f.close()
