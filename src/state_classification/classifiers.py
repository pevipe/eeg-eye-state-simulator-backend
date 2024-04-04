import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy import stats

from sklearn.model_selection import KFold, cross_val_score

classifier_dict = {'AB': AdaBoostClassifier(learning_rate=0.0645050649100435, n_estimators=30,
                                            random_state=3),
                   'DT': DecisionTreeClassifier(),
                   'KNN': KNeighborsClassifier(),
                   'LDA': LinearDiscriminantAnalysis(),
                   'LR': LogisticRegression(),
                   'RF': RandomForestClassifier(),
                   'QDA': QuadraticDiscriminantAnalysis(),
                   'SVM': SVC(C=1.2747788661328625, coef0=0.00425697976468159, random_state=0,
                              shrinking=False, tol=1.1901693317508151e-05)
                   }


class Classifier:
    def __init__(self, dataset):
        self.classifier = None
        self.x = dataset[:, 0:2]
        self.y = dataset[:, 2]
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=0)
        self.scores = None

    def __str__(self):
        if self.classifier is None:
            return "No classifier set"
        if self.scores is None:
            return "Classifier has not been executed yet"
        return ("Classifier: " + str(self.classifier) +
                "\nCross-validation scores: " + str(self.scores) +
                "\nAverage accuracy: " + str(self.scores.mean()) +
                "\nStandard deviation: " + str(self.scores.std()))

    def set_tree_classifier(self):
        self.classifier = DecisionTreeClassifier()

    def set_knn_classifier(self, n_neighbors=3):
        self.classifier = KNeighborsClassifier(n_neighbors)

    def set_svm_classifier(self, kernel='linear'):
        self.classifier = SVC(kernel=kernel)

    def classify(self):
        if self.classifier is None:
            print("No classifier set")
            return
        self.scores = cross_val_score(self.classifier, self.x, self.y, cv=self.kfold)


class AllClassifiers:
    def __init__(self, dataset, n_folds=10):
        # Set inputs and targets of classifiers
        self.x = dataset[:, 0:2]
        self.y = dataset[:, 2]

        # Set kfold and list of classiffiers
        self.kfold = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        self.classifiers = classifier_dict

        # Set empty list of scores
        self.scores = []

    def __str__(self):
        if not self.scores:
            return "Classifier has not been executed yet"
        result = ("##############################\n"
                  "# Results of all classifiers #\n"
                  "##############################\n")
        for name, results in self.scores:
            result += (name + "\nCross-validation scores: " + str(results)
                       + "\nAverage accuracy: " + str(results.mean())
                       + "\nStandard deviation: " + str(results.std()) + "\n\n")
        return result

    def preprocess(self):
        self.x = np.array([stats.zscore(self.x[:, 0]), stats.zscore(self.x[:, 1])]).T

    def classify(self, preprocess=False):
        self.preprocess() if preprocess else None
        for name in self.classifiers.keys():
            scores = cross_val_score(self.classifiers[name], self.x, self.y, cv=self.kfold)
            self.scores.append((name, scores))

    def save_results(self, route, synthesized=True, description=""):
        with open(route, 'w') as f:
            f.write(description + "\n")
            if synthesized:
                f.write("classifier,mean_accuracy,std_accuracy\n")
                for name, results in self.scores:
                    f.write(name + "," + str(results.mean()) + "," + str(results.std()) + "\n")
            else:
                f.write("classifier,1,2,3,4,5,6,7,8,9,10,mean_accuracy,std_accuracy\n")
                for name, results in self.scores:
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

        # Allow to take the classifiers with customized hyperparameter optimization
        if n_subject is not None and 0 < n_subject <= len(classifiers_per_subject):
            self.get_classifiers()
        else:
            print("No classifiers optimized for this subject. Using default hyperparameters.")
            self.classifiers = classifier_dict

        self.scores = []

    def get_classifiers(self):
        # csv will have: subject,classifier_name,string of classifier to create
        # Open a csv file and read the hyperparameters for each classifier
        data = np.loadtxt(self.customization_path, "rb", delimiter=",", skiprows=1)
        data = data[1:, :]  # From second row to the end
        classifiers = data[data[:, 0] == self.n_subject][:, 1:]
        for name, declaration in classifiers:
            self.classifiers[name] = eval(declaration)
