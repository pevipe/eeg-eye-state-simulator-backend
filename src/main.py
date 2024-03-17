from src.feature_extraction.load_data import load_dataset
from src.feature_extraction.constants import window_size
from src.state_classification.classifier import Classifier, TestClassifiers

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset(window_size)

    # # Create the classifier
    # classifier = Classifier(dataset)
    #
    # # Classify using decision tree
    # classifier.set_tree_classifier()
    # classifier.classify()
    # print(classifier)
    #
    # # Classify using knn
    # classifier.set_knn_classifier()
    # classifier.classify()
    # print(classifier)
    #
    # # Classify using svm
    # classifier.set_svm_classifier()
    # classifier.classify()
    # print(classifier)
    #
    # # Classify using knn with 5 neighbors
    # classifier.set_knn_classifier(5)
    # classifier.classify()
    # print(classifier)

    all_classifiers = TestClassifiers(dataset)
    all_classifiers.classify()
    print(all_classifiers)
