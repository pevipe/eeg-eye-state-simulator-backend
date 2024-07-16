import shutil

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from hpsklearn import HyperoptEstimator, any_preprocessing, quadratic_discriminant_analysis
from hyperopt import tpe
from hpsklearn import (svc, decision_tree_classifier, k_neighbors_classifier, random_forest_classifier,
                       linear_discriminant_analysis, ada_boost_classifier)

classifier_dict = {
    'AdaBoost': ada_boost_classifier('abc'),
    'DecisionTree': decision_tree_classifier('dtc'),
    'kNN': k_neighbors_classifier('knc'),
    'LDA': linear_discriminant_analysis('lda'),
    'RandomForest': random_forest_classifier('rfc'),
    'QDA': quadratic_discriminant_analysis('qda'),
    'SVM': svc('svc'),
}


class SingleOptimizer:
    def __init__(self, dataset, classifier, general_optimization_loc, output_path):
        self.dataset = dataset
        self.classifier = classifier
        self.general_loc = general_optimization_loc
        self.output_path = output_path
        self.model = self.optimize()
        self.save_results()

    def optimize(self):
        x = self.dataset[:, 0:2].astype('float32')
        y = LabelEncoder().fit_transform(self.dataset[:, 2].astype('str'))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)

        cl = classifier_dict[self.classifier]
        model = HyperoptEstimator(classifier=cl, preprocessing=any_preprocessing('pre'),
                                  algo=tpe.suggest, max_evals=50, trial_timeout=30)
        try:
            # perform the search
            model.fit(x_train, y_train)
            # summarize performance
            acc = model.score(x_test, y_test)
            print("Accuracy: " + str(acc), end="\n\n")
            return model.best_model()
        except Exception as e:
            return None

    def save_results(self):
        if self.model is None:
            shutil.copyfile(self.general_loc, self.output_path)
        else:
            with open(self.output_path, "w") as f:
                f.write((str(self.model['learner']) + ";" +
                         str(self.model['preprocs'])).replace("\n", "").replace(" ", ""))

