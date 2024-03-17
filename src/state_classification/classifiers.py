from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold, cross_val_score


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


class TestClassifiers:
    def __init__(self, dataset):
        # Set inputs and targets of classifiers
        self.x = dataset[:, 0:2]
        self.y = dataset[:, 2]

        # Set kfold and list of classiffiers
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=0)
        self.classifiers = [
            ('LR', LogisticRegression()),
            ('KNN', KNeighborsClassifier()),
            ('CART', DecisionTreeClassifier()),
            ('LDA', LinearDiscriminantAnalysis()),
            ('NB', GaussianNB()),
            ('SVM', SVC(gamma='auto'))
        ]
        self.scores = []

    def __str__(self):
        if not self.scores:
            return "Classifier has not been executed yet"
        result = ""
        for name, results in self.scores:
            result += (name + "\nCross-validation scores: " + str(results)
                       + "\nAverage accuracy: " + str(results.mean())
                       + "\nStandard deviation: " + str(results.std()) + "\n\n")
        return result

    def classify(self):
        for name, classifier in self.classifiers:
            scores = cross_val_score(classifier, self.x, self.y, cv=self.kfold)
            self.scores.append((name, scores))

