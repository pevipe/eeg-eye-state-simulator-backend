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

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

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


class AllClassifiers:
    def __init__(self, dataset, n_folds=10):
        # Set inputs and targets of classifiers
        self.x = dataset[:, 0:2]
        self.y = dataset[:, 2]

        # Set kfold and list of classiffiers
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        self.classifiers = classifier_dict

        # Set empty list of scores
        self.scores = {}

    def __str__(self):
        if not self.scores:
            return "Classifier has not been executed yet"
        result = ("##############################\n"
                  "# Results of all classifiers #\n"
                  "##############################\n")
        for name in self.scores.keys():
            results = self.scores[name]
            result += (name + "\nCross-validation scores: " + str(results)
                       + "\nAverage accuracy: " + str(results.mean())
                       + "\nStandard deviation: " + str(results.std()) + "\n\n")
        return result

    def preprocess(self):
        self.x = np.array([stats.zscore(self.x[:, 0]), stats.zscore(self.x[:, 1])]).T

    def get_cross_val_scores(self, preprocess=False):
        self.preprocess() if preprocess else None
        for name in self.classifiers.keys():
            scores = cross_val_score(self.classifiers[name], self.x, self.y, cv=self.kfold)
            self.scores[name] = scores

    def save_cross_val_results(self, route, synthesized=True, description=""):
        # Avoid errors with nonexistent dirs
        if not os.path.exists(os.path.dirname(route)):
            os.makedirs(os.path.dirname(route))

        with open(route, 'w') as f:
            f.write(description + "\n")
            if synthesized:
                f.write("classifier,mean_accuracy,std_accuracy\n")
                for name in self.scores.keys():
                    results = self.scores[name]
                    f.write(name + "," + str(results.mean()) + "," + str(results.std()) + "\n")
            else:
                f.write("classifier,1,2,3,4,5,6,7,8,9,10,mean_accuracy,std_accuracy\n")
                for name in self.scores.keys():
                    results = self.scores[name]
                    f.write(name + ",")
                    for fold_accuracy in results:
                        f.write(str(fold_accuracy) + ",")
                    f.write(str(results.mean()) + "," + str(results.std()) + "\n")
        f.close()


class CustomizedClassifiers(AllClassifiers):
    def __init__(self, dataset, customization_path, n_subject=None, n_folds=10):
        super().__init__(dataset, n_folds)
        self.n_subject = n_subject
        self.customization_path = customization_path
        self.classifiers = {}
        self.preprocessers = {}

        # Allow to take the classifiers with customized hyperparameter optimization
        if n_subject is not None and 0 < n_subject:
            self.get_classifiers()
        else:
            print("No classifiers optimized for this subject. Using default hyperparameters.")
            self.classifiers = classifier_dict

        self.scores = {}

    def get_classifiers(self):
        self.classifiers = {}  # Reset the classifiers
        # csv will have: subject,classifier_name,classifier_init,preprocess_init,accuracy
        # Open a csv file and read the hyperparameters for each classifier
        data = np.loadtxt(self.customization_path, dtype=str, delimiter=";", skiprows=1)
        # each row of data has the following shape: [[n_subject] model_name, model_init, preprocess_init]
        if self.n_subject is None:
            models_for_this_subject = data[:, :3]
        else:
            models_for_this_subject = data[data[:, 0] == str(self.n_subject)][:, 1:4]  # Get the models for this subject
        for (name, classifier, preprocesser) in models_for_this_subject:
            self.classifiers[name] = eval(classifier)
            self.preprocessers[name] = eval(preprocesser)

    def get_cross_val_scores(self, preprocess=True):
        for name in self.classifiers.keys():
            pipe_tuple = self.preprocessers[name] + (self.classifiers[name],)
            pipe = make_pipeline(*pipe_tuple)
            scores = cross_val_score(pipe, self.x, self.y, scoring='accuracy', cv=self.kfold)
            self.scores[name] = scores

    def get_classification_report(self):
        for name in self.classifiers.keys():
            pipe_tuple = self.preprocessers[name] + (self.classifiers[name],)
            pipe = make_pipeline(*pipe_tuple)
            try:
                predictions = cross_val_predict(pipe, self.x, self.y, cv=self.kfold, method='predict')
                report = classification_report(self.y, predictions, output_dict=True)
            except Exception as e:
                print("Error while evaluating subject " + str(self.n_subject) + ":" + str(e))
                report = {"0.0": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
                          "1.0": {"precision": 0, "recall": 0, "f1-score": 0, "support": 0},
                          "accuracy": 0}
            self.scores[name] = report

    def save_report_results(self, route):
        # Avoid errors with nonexistent dirs
        if not os.path.exists(os.path.dirname(route)):
            os.makedirs(os.path.dirname(route))

        with open(route, 'w') as f:

            f.write("classifier,precision_0,f1_0,precision_1,f1_1,accuracy\n")

            for name in self.scores.keys():
                report = self.scores[name]
                precision_0 = str(round(report['0.0']['precision'], 3))
                precision_1 = str(round(report['1.0']['precision'], 3))
                f1_0 = str(round(report['0.0']['f1-score'], 3))
                f1_1 = str(round(report['1.0']['f1-score'], 3))
                accuracy = str(round(report['accuracy'], 3))

                f.write(name + "," + precision_0 + "," + f1_0 + "," + precision_1 + "," + f1_1 + "," + accuracy + "\n")
            f.close()
