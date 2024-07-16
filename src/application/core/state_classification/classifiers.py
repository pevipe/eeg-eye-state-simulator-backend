import os

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import stats

from sklearn.preprocessing import (Binarizer, MinMaxScaler, MaxAbsScaler, Normalizer, RobustScaler, StandardScaler,
                                   QuantileTransformer, PowerTransformer, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, SplineTransformer, KBinsDiscretizer)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score

classifier_dict = {'AB': AdaBoostClassifier(learning_rate=0.0645050649100435, n_estimators=30,
                                            random_state=3),
                   'DT': DecisionTreeClassifier(),
                   'KNN': KNeighborsClassifier(),
                   'LDA': LinearDiscriminantAnalysis(),
                   'RF': RandomForestClassifier(),
                   'QDA': QuadraticDiscriminantAnalysis(),
                   'SVM': SVC(C=1.2747788661328625, coef0=0.00425697976468159, random_state=0,
                              shrinking=False, tol=1.1901693317508151e-05)
                   }


class CustomizedClassifier:
    def __init__(self, dataset_path, customization_path, output_path, exact_index_path, train_set_size):
        # Init variables
        self.total_accuracy, self.precision_opened, self.precision_closed = None, None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.pipeline = None
        self.train_set_size = train_set_size

        self.get_dataset(dataset_path)
        self.get_pipeline(customization_path)
        self.predictions = self.get_prediction()
        self.results = self.get_results(self.predictions, exact_index_path)
        self.save_results(output_path)

    def get_dataset(self, dataset_path):
        d = np.loadtxt(dataset_path, delimiter=",")
        x = d[:, 0:2]
        y = d[:, 2]
        # Take the middle windows for testing, and the rest for training the classifier
        division_index_left = int(len(x) * self.train_set_size / 200)
        test_size = len(x) - division_index_left*2
        division_index_right = division_index_left + test_size
        self.X_train, self.X_test = np.concatenate((x[:division_index_left], x[division_index_right:])), x[division_index_left:division_index_right]
        self.y_train, self.y_test = np.concatenate((y[:division_index_left], y[division_index_right:])), y[division_index_left:division_index_right]
        # Take the last n windows for testing, and the rest for training the classifier
        # division_index = len(x) - int(len(x) * self.train_set_size / 100)
        # self.X_test, self.X_train = x[:division_index], x[division_index:]
        # self.y_test, self.y_train = y[:division_index], y[division_index:]

    def get_pipeline(self, customization_path):
        command = np.loadtxt(customization_path, dtype=str, delimiter=";")
        preprocesser = eval(command[1])
        classifier = eval(command[0])
        pipe_tuple = preprocesser + (classifier,)
        self.pipeline = make_pipeline(*pipe_tuple)

    def get_prediction(self):
        # Train the classifier
        self.pipeline.fit(self.X_train, self.y_train)

        # Use classifier on the test set
        y_pred = self.pipeline.predict(self.X_test)

        return y_pred

    def get_results(self, predictions, exact_index_path):
        exact_indexes = np.loadtxt(exact_index_path, delimiter=",")
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

    def save_results(self, output_path):
        # Avoid errors with nonexistent dirs
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        with open(output_path, 'w') as f:
            f.write("precision_closed;precision_opened;accuracy;real_tags_in_test;predicted_tags_in_test\n")
            f.write(str(self.precision_closed) + ";" + str(self.precision_opened) + ";" + str(self.total_accuracy) +
                    ";" + str(self.y_test) + ";" + str(self.predictions) + ";" + "\n")
            f.close()

