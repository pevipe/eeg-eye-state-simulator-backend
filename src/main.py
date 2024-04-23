import os
import time

from src.feature_extraction.data_loaders import ProvidedDatasetLoader, DHBWDatasetLoader, \
    ProvidedDatasetIndividualLoader
from src.feature_extraction.constants import window_size
from src.state_classification.classifiers import AllClassifiers, CustomizedClassifiers
from src.state_classification.hyperparameter_optimization import ClassifierOptimization

from run_algorithms import individualized, all_subjects_together, transform_results, run_all, get_routes


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
    all_classifiers.get_cross_val_scores()
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
    all_classifiers.get_cross_val_scores()
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
        all_classifiers.get_cross_val_scores()
        all_classifiers.save_cross_val_results("../out/results/individual/subject_" + str(i + 1) + "_complete.csv",
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
output_path_from_configuration = {'00': "../out/results/differentiated/full_results_hyperopt_00",
                                  '01': "../out/results/differentiated/full_results_hyperopt_01",
                                  '10': "../out/results/differentiated/full_results_hyperopt_10",
                                  '11': "../out/results/differentiated/full_results_hyperopt_11"}
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
    for i, subject in enumerate(provided_dataset):
        all_classifiers = CustomizedClassifiers(subject, hyperparams_path, n_subject=i + 1)
        all_classifiers.get_classification_report()
        # all_classifiers.save_report_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")
        # print("Subject " + str(i + 1) + " finished.")
        # print(all_classifiers, end="\n\n")


def do_dataset_1_with_optimizations_differentiating_pure_windows(w_size):
    out_dataset_path = "../out/datasets/individual_pure_windows/size_"+str(w_size)
    hyperparams_path = "../out/results/opt_hyperopt_pure_windows_size_" + str(w_size) + "/optimized_hyperparameters.csv"
    output_path = "../out/results/differentiated/full_results_hyperopt_size_" + str(w_size)

    # Load dataset
    dataset_loader = ProvidedDatasetIndividualLoader("../data/dataset_1", out_dataset_path,
                                                     200, w_size)
    dataset_loader.load_all_datasets(overlap=w_size-2, normalize=False, pure_windows=True)
    provided_dataset = dataset_loader.dataset
    print("*************************\n" +
          "* DATASET 1 (optimized) *\n" +
          "*************************\n")

    if not os.path.exists(os.path.dirname(hyperparams_path)):
        os.mkdir(os.path.dirname(hyperparams_path))
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))

    for i, subject in enumerate(provided_dataset):
        # Optimize hyperparameters

        print("*** Optimizing hyperparameters for subject " + str(i + 1) + " ***\n")
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(subject, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results(
            "../out/results/opt_hyperopt_pure_windows_size_" + str(w_size) + "/optimized_hyperparameters.csv")
        print("Subject " + str(i + 1) + " optimized in " + str(time.time() - start_time) + " seconds.\n")

    for i, subject in enumerate(provided_dataset):
        # Load custom classifiers
        all_classifiers = CustomizedClassifiers(subject, hyperparams_path, n_subject=i + 1)
        all_classifiers.get_classification_report()
        all_classifiers.save_report_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")
        print("Subject " + str(i + 1) + " finished.")


if __name__ == '__main__':
    # for config in ['00', '01', '10', '11']:
    #     do_dataset_1_with_optimizations(config)

    # for w in [8, 5]:
    # do_dataset_1_with_optimizations_differentiating_pure_windows(8)

    # pure_windows_single_subject("../data/dataset_1", "../out/datasets/individual_pure_windows/size_10",
    #                             "../out/results/opt_hyperopt_pure_windows_size_10/other.csv",
    #                             "../out/results/differentiated/full_results_hyperopt_size_10_new/subject_7.csv", 10, 7)

    # # Single optimization and training for all the subjects
    # all_subjects_together("../data/dataset_1", "../out/results_collective/hyperparameters_not_pure.csv",
    #                       "../out/results_collective/not_pure", False)
    # all_subjects_together("../data/dataset_1", "../out/results_collective/hyperparameters_pure.csv",
    #                       "../out/results_collective/pure", True)

    # # Individual optimization and training for each of the subjects
    # for config in [10, 8, 5]:
    #     individualized("../data/dataset_1", "../out/datasets/individual_running",
    #                    "../out/hyperparameter_optimizations/individual/" + str(config) + "s/all_windows.csv",
    #                    "../out/results/individual/" + str(config) + "s/not_pure", config, False)
    #     transform_results("../out/results/individual/" + str(config) + "s/not_pure", False)

    # transform results to single file
    # transform_results("../out/results_collective/not_pure", False)
    # transform_results("../out/results_collective/pure", True)

    # for config in [10, 8, 5]:
    #     transform_results("../out/results/individual/" + str(config) + "s/not_pure", False)
    run_all(True, 10, True, "../data/dataset_1")
    # print(get_routes('datasets', True, 10, True))

