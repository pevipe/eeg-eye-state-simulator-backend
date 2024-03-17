from src.feature_extraction.data_loaders import ProvidedDatasetLoader, DHBWDatasetLoader
from src.feature_extraction.constants import window_size
from src.state_classification.classifiers import TestClassifiers

if __name__ == '__main__':
    # Load first dataset
    provided_dataset_loader = ProvidedDatasetLoader("../data/dataset_1", 200, window_size)
    provided_dataset_loader.load_dataset()
    provided_dataset = provided_dataset_loader.dataset

    # Load second dataset
    dhbw_dataset_loader = DHBWDatasetLoader("../data/dataset_2/eeg-eye-state.csv", 128, window_size, 117)
    dhbw_dataset_loader.load_dataset()
    dhbw_dataset = dhbw_dataset_loader.dataset

    # Test classifiers
    print("*****************\n" +
          "*** DATASET 1 ***\n" +
          "*****************\n")
    all_classifiers = TestClassifiers(provided_dataset)
    all_classifiers.classify()
    print(all_classifiers, end="\n\n")

    print("*****************\n" +
          "*** DATASET 2 ***\n" +
          "*****************\n")
    all_classifiers = TestClassifiers(dhbw_dataset)
    all_classifiers.classify()
    print(all_classifiers)
