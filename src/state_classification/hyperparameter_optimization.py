from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator, any_preprocessing, quadratic_discriminant_analysis
from hyperopt import tpe
from hpsklearn import (svc, decision_tree_classifier, k_neighbors_classifier, random_forest_classifier,
                       linear_discriminant_analysis, ada_boost_classifier)

classifier_dict = {
    'AdaBoost': ada_boost_classifier('abc'),
    'DecisionTree': decision_tree_classifier('dtc'),
    'KNN': k_neighbors_classifier('knc'),
    'LDA': linear_discriminant_analysis('lda'),
    # 'LogisticRegression': TODO: DELETE
    'RandomForest': random_forest_classifier('rfc'),
    'QDA': quadratic_discriminant_analysis('qda'),
    'SVM': svc('svc'),
}


class ClassifierOptimization:
    def __init__(self, dataset, n_subject=None):
        # Set inputs and targets of classifiers
        self.x = dataset[:, 0:2].astype('float32')
        self.y = LabelEncoder().fit_transform(dataset[:, 2].astype('str'))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.33,
                                                                                random_state=1)

        # Set list of classiffiers
        self.classifiers = classifier_dict
        self.scores = []
        self.n_subject = n_subject

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

    def save_results(self, file):
        with open(file, "a+") as f:
            if self.n_subject is not None:
                f.seek(0)
                first_line = f.readline()
                if first_line.startswith("subject"):
                    f.seek(0, 2)  # Move cursor to the end of the file
                else:
                    f.seek(0)  # Remove content and write header
                    f.truncate()
                    f.write("subject;classifier;model;preprocessing;accuracy\n")
                for name, accuracy, model in self.scores:
                    f.write((str(self.n_subject) + ";" + name + ";" + str(model['learner']) +
                            ";" + str(model['preprocs']) + ";" + str(round(accuracy, 3)) + "\n").replace("\n   ", ""))
            else:
                if f.readline().startswith("classifier"):
                    f.seek(0, 2)
                else:
                    f.seek(0)
                    f.truncate()
                    f.write("classifier;model;preprocessing;accuracy\n")
                for name, accuracy, model in self.scores:
                    f.write(name + ";" + str(model['learner']) + ";" + str(model['preprocs']) +
                            ";" + str(round(accuracy, 3)) + "\n")
