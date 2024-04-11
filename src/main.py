import os
import time

from src.feature_extraction.data_loaders import ProvidedDatasetLoader, DHBWDatasetLoader, \
    ProvidedDatasetIndividualLoader
from src.feature_extraction.constants import window_size
from src.state_classification.classifiers import AllClassifiers, CustomizedClassifiers
from src.state_classification.hyperparameter_optimization import ClassifierOptimization


######################################
# BASIC OPERATIONS (useless for now) #
######################################
def do_dataset_1():
    # Load first dataset
    provided_dataset_loader = ProvidedDatasetLoader("../data/dataset_1", 200, window_size)
    provided_dataset_loader.load_dataset()
    provided_dataset = provided_dataset_loader.dataset

    # Test classifiers
    print("*****************\n" +
          "*** DATASET 1 ***\n" +
          "*****************\n")
    all_classifiers = AllClassifiers(provided_dataset)
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
    all_classifiers = AllClassifiers(dhbw_dataset)
    all_classifiers.classify()
    print(all_classifiers)


def do_dataset_1_one_at_a_time():
    # Load datasets from individual subjects
    provided_dataset_individual_loader = ProvidedDatasetIndividualLoader("../data/dataset_1",
                                                                         "../out/datasets/individual", 200, window_size)
    provided_dataset_individual_loader.load_all_datasets(overlap=8)
    provided_datasets_individual = provided_dataset_individual_loader.dataset

    print("**************************\n" +
          "* DATASET 1 (individual) *\n" +
          "**************************\n")
    for i, dataset in enumerate(provided_datasets_individual):
        all_classifiers = AllClassifiers(dataset)
        all_classifiers.preprocess()
        all_classifiers.classify()
        all_classifiers.save_results("../out/results/individual/subject_" + str(i + 1) + "_complete.csv",
                                     synthesized=False, description="Results from subject " + str(i + 1))
        print("Subject " + str(i + 1) + ":")
        print(all_classifiers, end="\n\n")


# TODO: parametrizar y exportar a otro fichero
###############################
# HYPERPARAMETER OPTIMIZATION #
###############################
def optimize_hyperparameters_from_dataset_1():
    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1",
                                                     "../out/datasets/individual_without_normalization",
                                                     200, window_size)
    dataset_loader.load_all_datasets(overlap=8, normalize=False, pure_windows=False)
    provided_dataset = dataset_loader.dataset

    # Optimize hyperparameters
    for i, dataset in enumerate(provided_dataset):
        print("\n******************************************************\n"
              "Optimizing hyperparameters for subject " + str(i + 1))
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(dataset, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results("../out/results/opt_hyperopt/optimized_hyperparameters.csv")
        print("Subject " + str(i + 1) + " optimized in " + str(start_time - time.time()) + " seconds.\n")


def optimize_hyperparameters_from_dataset_1_with_avg_diff():
    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1",
                                                     "../out/datasets/individual_strange_normalization",
                                                     200, window_size)
    dataset_loader.load_all_datasets(overlap=8, normalize=True, pure_windows=False)
    provided_dataset = dataset_loader.dataset

    # Optimize hyperparameters
    for i, dataset in enumerate(provided_dataset):
        print("\n******************************************************\n"
              "Optimizing hyperparameters for subject " + str(i + 1))
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(dataset, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results("../out/results/opt_hyperopt_with_avg_diff/optimized_hyperparameters.csv")
        print("Subject " + str(i + 1) + " optimized in " + str(start_time - time.time()) + " seconds.\n")


def optimize_hyperparameters_from_dataset_1_pure_windows():
    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1",
                                                     "../out/datasets/individual_pure_windows", 200, window_size)
    dataset_loader.load_all_datasets(overlap=8, normalize=False, pure_windows=True)
    provided_dataset = dataset_loader.dataset

    # Optimize hyperparameters
    for i, dataset in enumerate(provided_dataset):
        print("\n******************************************************\n"
              "Optimizing hyperparameters for subject " + str(i + 1))
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(dataset, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results("../out/results/opt_hyperopt_pure_windows/optimized_hyperparameters.csv")
        print("Subject " + str(i + 1) + " optimized in " + str(start_time - time.time()) + " seconds.\n")


def optimize_hyperparameters_from_dataset_1_pure_windows_norm():
    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1",
                                                     "../out/datasets/individual_pure_windows_norm", 200, window_size)
    dataset_loader.load_all_datasets(overlap=8, normalize=True, pure_windows=True)
    provided_dataset = dataset_loader.dataset

    # Optimize hyperparameters
    for i, dataset in enumerate(provided_dataset):
        print("\n******************************************************\n"
              "Optimizing hyperparameters for subject " + str(i + 1))
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(dataset, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results(
            "../out/results/opt_hyperopt_pure_windows_norm/optimized_hyperparameters.csv")
        print("Subject " + str(i + 1) + " optimized in " + str(start_time - time.time()) + " seconds.\n")


def setup_optimization():
    file_list = [
        "../out/results/opt_hyperopt_wo_avg_diff/optimized_hyperparameters.csv",
        "../out/results/opt_hyperopt_with_avg_diff/optimized_hyperparameters.csv",
        "../out/results/opt_hyperopt_pure_windows/optimized_hyperparameters.csv",
        "../out/results/opt_hyperopt_pure_windows_norm/optimized_hyperparameters.csv"
    ]
    for file in file_list:
        if os.path.exists(file):
            os.remove(file)
        os.mkdir(os.path.dirname(file))


out_dataset_from_configuration = {'00': "../out/datasets/individual_without_normalization",
                                  '01': "../out/datasets/individual_strange_normalization",
                                  '10': "../out/datasets/individual_pure_windows",
                                  '11': "../out/datasets/individual_pure_windows_norm"}
hyperaparams_from_configuration = {'00': "../out/results/opt_hyperopt_wo_avg_diff/optimized_hyperparameters.csv",
                                   '01': "../out/results/opt_hyperopt_with_avg_diff/optimized_hyperparameters.csv",
                                   '10': "../out/results/opt_hyperopt_pure_windows/optimized_hyperparameters.csv",
                                   '11': "../out/results/opt_hyperopt_pure_windows_norm/optimized_hyperparameters.csv"}
output_path_from_configuration = {'00': "../out/results/full_results_hyperopt_00",
                                  '01': "../out/results/full_results_hyperopt_01",
                                  '10': "../out/results/full_results_hyperopt_10",
                                  '11': "../out/results/full_results_hyperopt_11"}
dataset_load_from_configuration = {
    '00': "dataset_loader.load_all_datasets(overlap=8, normalize=False, pure_windows=False)",
    '01': "dataset_loader.load_all_datasets(overlap=8, normalize=True, pure_windows=False)",
    '10': "dataset_loader.load_all_datasets(overlap=8, normalize=False, pure_windows=True)",
    '11': "dataset_loader.load_all_datasets(overlap=8, normalize=True, pure_windows=True)"}


def do_dataset_1_with_optimizations(configuration):
    out_dataset_path = out_dataset_from_configuration[configuration]
    hyperparams_path = hyperaparams_from_configuration[configuration]
    output_path = output_path_from_configuration[configuration]

    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1", out_dataset_path,
                                                     200, window_size)

    eval(dataset_load_from_configuration[configuration])
    provided_dataset = dataset_loader.dataset

    print("****************************\n" +
          "* DATASET 1 (optimized) "+configuration+" * \n" +
          "****************************\n")
    for i, dataset in enumerate(provided_dataset):
        all_classifiers = CustomizedClassifiers(dataset, hyperparams_path, n_subject=i + 1)
        all_classifiers.classify()
        all_classifiers.save_results(output_path + "/subject_" + str(i + 1) + "_complete.csv",
                                     synthesized=False, description="Results from subject " + str(i + 1))
        print("Subject " + str(i + 1) + ":")
        print(all_classifiers, end="\n\n")


if __name__ == '__main__':
    for config in ['01', '10', '11']:
        do_dataset_1_with_optimizations(config)
