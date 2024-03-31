# example of hyperopt-sklearn for the sonar classification dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator, any_classifier, any_preprocessing
from hyperopt import tpe
from hpsklearn import (svc, decision_tree_classifier, k_neighbors_classifier, random_forest_classifier, linear_svc,
                       linear_discriminant_analysis, ada_boost_classifier, gradient_boosting_classifier)


class TestOptimizedClassifiers:
    def __init__(self, dataset):
        # Set inputs and targets of classifiers
        self.x = dataset[:, 0:2].astype('float32')
        self.y = LabelEncoder().fit_transform(dataset[:, 2].astype('str'))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.33,
                                                                                random_state=1)

        # Set list of classiffiers
        self.classifiers = {
            'AdaBoost': ada_boost_classifier('abc'),
            'DecisionTree': decision_tree_classifier('dtc'),
            'GradientBoosting': gradient_boosting_classifier('gbc'),
            'KNeighbors': k_neighbors_classifier('knc'),
            'LinearSVC': linear_svc('lsvc'),
            'LinearDiscriminantAnalysis': linear_discriminant_analysis('lda'),
            'RandomForest': random_forest_classifier('rfc'),
            'SVC': svc('svc')
        }
        self.scores = []

    def try_all(self):
        for name in self.classifiers.keys():
            classifier = self.classifiers[name]
            print("Classifier: " + name)

            model = HyperoptEstimator(classifier=classifier, preprocessing=any_preprocessing('pre'),
                                      algo=tpe.suggest, max_evals=50, trial_timeout=30)
            # perform the search
            model.fit(self.X_train, self.y_train)
            # summarize performance
            acc = model.score(self.X_test, self.y_test)
            print("Accuracy: " + str(acc), end="\n\n")
            self.scores.append((name, acc, model.best_model()))

    def try_one(self, cl):
        # print("Classifier: " + self.classifiers[cl])
        name, classifier = cl, self.classifiers[cl]
        print("Classifier: " + name)

        model = HyperoptEstimator(classifier=classifier, preprocessing=any_preprocessing('pre'),
                                  algo=tpe.suggest, max_evals=50, trial_timeout=30)
        # perform the search
        model.fit(self.X_train, self.y_train)
        # summarize performance
        acc = model.score(self.X_test, self.y_test)
        print("Accuracy: " + str(acc), end="\n\n")
        self.scores.append((name, acc, model.best_model()))

    def save_results(self, file, description=None):
        with open(file, "w") as f:
            if description is not None:
                f.write(description + "\n")
            f.write("classifier,accuracy,model\n")
            for name, accuracy, model in self.scores:
                f.write(name+","+str(round(accuracy, 3))+","+str(model)+"\n")

