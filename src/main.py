from src.feature_extraction.data_loaders import ProvidedDatasetLoader, DHBWDatasetLoader, \
    ProvidedDatasetIndividualLoader
from src.feature_extraction.constants import window_size
from src.state_classification.classifiers import TestClassifiers
from src.state_classification.hyperparameter_optimization import ClassifierOptimization

def do_dataset_1():
    # Load first dataset
    provided_dataset_loader = ProvidedDatasetLoader("../data/dataset_1", 200, window_size)
    provided_dataset_loader.load_dataset()
    provided_dataset = provided_dataset_loader.dataset

    # Test classifiers
    print("*****************\n" +
          "*** DATASET 1 ***\n" +
          "*****************\n")
    all_classifiers = TestClassifiers(provided_dataset)
    all_classifiers.classify()
    print(all_classifiers, end="\n\n")


def do_dataset_2():
    # Load second dataset
    dhbw_dataset_loader = DHBWDatasetLoader("../data/dataset_2/eeg-eye-state.csv", 128, window_size, 117)
    dhbw_dataset_loader.load_dataset()
    dhbw_dataset = dhbw_dataset_loader.dataset

    # Classify and print results
    print("*****************\n" +
          "*** DATASET 2 ***\n" +
          "*****************\n")
    all_classifiers = TestClassifiers(dhbw_dataset)
    all_classifiers.classify()
    print(all_classifiers)


def do_dataset_1_one_at_a_time():
    # Load datasets from individual subjects
    provided_dataset_individual_loader = ProvidedDatasetIndividualLoader("../data/dataset_1", 200, window_size)
    provided_dataset_individual_loader.load_all_datasets(overlap=8)
    provided_datasets_individual = provided_dataset_individual_loader.dataset

    print("**************************\n" +
          "* DATASET 1 (individual) *\n" +
          "**************************\n")
    for i, dataset in enumerate(provided_datasets_individual):
        all_classifiers = TestClassifiers(dataset)
        all_classifiers.classify()
        all_classifiers.save_results("../out/results/individual/subject_" + str(i + 1) + "_complete.csv",
                                     synthetized=False, description="Results from subject " + str(i + 1))
        print("Subject " + str(i + 1) + ":")
        print(all_classifiers, end="\n\n")


def do_dataset_1_one_at_a_time_optimized():
    # Load datasets from individual subjects
    provided_dataset_individual_loader = ProvidedDatasetIndividualLoader("../data/dataset_1", 200, window_size)
    provided_dataset_individual_loader.load_all_datasets(overlap=8)
    provided_datasets_individual = provided_dataset_individual_loader.dataset

    print("**************************\n" +
          "* DATASET 1 (individual) *\n" +
          "**************************\n")
    for i, dataset in enumerate(provided_datasets_individual):
        optimized_classifiers = ClassifierOptimization(dataset, n_subject=i+1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results("../out/results/opt_hyperopt/subject_" + str(i + 1) + "_optimized.csv")


if __name__ == '__main__':
    do_dataset_1_one_at_a_time_optimized()
