import os
import time

import pandas as pd

from src.feature_extraction.data_loaders import ProvidedDatasetIndividualLoader, ProvidedDatasetLoader
from src.state_classification.classifiers import CustomizedClassifiers
from src.state_classification.hyperparameter_optimization import ClassifierOptimization


def get_routes(ty, individual, win_size, pure_windows):
    if ty == 'results' or ty == 'hyperparams' or ty == 'datasets':
        if individual:
            return f"../out/{ty}/individual/{win_size}s/{'pure_windows' if pure_windows else 'all_windows'}"
        else:
            if pure_windows:
                return f"../out/{ty}/collective/{win_size}s/{'pure_windows' if pure_windows else 'all_windows'}"
    else:
        raise Exception("Invalid type. It must be one of the following: 'results', 'hyperparams', 'datasets'")


def get_dataset(dataset_input_path, individual, win_size, pure_windows, dataset_output_path=None):
    if dataset_output_path is None:
        dataset_output_path = get_routes('datasets', individual, win_size, pure_windows)

    if individual:
        dataset_loader = ProvidedDatasetIndividualLoader(dataset_input_path, dataset_output_path, 200, win_size)
        dataset_loader.load_all_datasets(overlap=win_size - 2, normalize=False, pure_windows=pure_windows)
    else:
        dataset_loader = ProvidedDatasetLoader(dataset_input_path, 200, 10)
        dataset_loader.load_dataset(overlap=win_size - 2, pure_windows=pure_windows)

    return dataset_loader.dataset


def get_optimized_hyperparameters_for_all_subjects(dataset, hyperparams_output_path):
    total_time = 0

    for i, subject in enumerate(dataset):
        print("*** Optimizing hyperparameters for subject " + str(i + 1) + " ***\n")
        start_time = time.time()
        optimized_classifiers = ClassifierOptimization(subject, n_subject=i + 1)
        optimized_classifiers.try_all()
        optimized_classifiers.save_results(hyperparams_output_path)
        print("Subject " + str(i + 1) + " optimized in " + str(time.time() - start_time) + " seconds.\n")
        total_time += time.time() - start_time

    print("\nOptimization finished in " + str(total_time) + " seconds.\n")


def get_optimized_hyperparameters_for_one_subject(dataset, hyperparams_output_path, n_subject):

    print("*** Optimizing hyperparameters for subject " + str(n_subject) + " ***\n")
    start_time = time.time()
    optimized_classifiers = ClassifierOptimization(dataset[n_subject-1], n_subject=n_subject)
    optimized_classifiers.try_all()
    optimized_classifiers.save_results(hyperparams_output_path)
    print("Subject " + str(n_subject) + " optimized in " + str(time.time() - start_time) + " seconds.\n")


def get_optimized_hyperparameters_for_all_subjects_one_optimization(dataset, optimized_hyperparameters_path):
    print("*** Optimizing hyperparameters for all subjects ***\n")
    start_time = time.time()
    optimizer = ClassifierOptimization(dataset)
    optimizer.try_all()
    optimizer.save_results(optimized_hyperparameters_path)
    print("Subjects optimized in " + str(time.time() - start_time) + " seconds.\n")


def get_estimated_performance(dataset, hyperparam_path, output_path, pure_windows, n_subject=None):
    if n_subject is None:
        for i, subject in enumerate(dataset):
            print("*** Classifying subject " + str(i + 1) + " ***\n")
            start_time = time.time()
            all_classifiers = CustomizedClassifiers(subject, hyperparam_path, n_subject=i + 1)

            if pure_windows:
                all_classifiers.get_classification_report()
                all_classifiers.save_report_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")
            else:
                all_classifiers.get_cross_val_scores()
                all_classifiers.save_cross_val_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")

            print("Subject " + str(i + 1) + " classified in " + str(time.time() - start_time) + " seconds.\n")
    else:
        print("*** Classifying subject " + str(n_subject) + " ***\n")
        start_time = time.time()

        all_classifiers = CustomizedClassifiers(dataset[n_subject - 1], hyperparam_path, n_subject=n_subject)
        if pure_windows:
            all_classifiers.get_classification_report()
            all_classifiers.save_report_results(output_path + "/subject_" + str(n_subject) + "_complete.csv")
        else:
            all_classifiers.get_cross_val_scores()
            all_classifiers.save_cross_val_results(output_path + "/subject_" + str(n_subject) + "_complete.csv")

        print("Subject " + str(n_subject) + " classified in " + str(time.time() - start_time) + " seconds.\n")


def get_estimated_performance_one_optimization(dataset, optimized_hyperparameters_path, output_path, pure_windows):

    for i in range(7):
        print("*** Classifying subject " + str(i+1) + " ***\n")
        start_time = time.time()

        all_classifiers = CustomizedClassifiers(dataset[i], optimized_hyperparameters_path)
        all_classifiers.get_classifiers()  # Has to be manually called, as n_subject is None

        if pure_windows:
            all_classifiers.get_classification_report()
            all_classifiers.save_report_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")
        else:
            all_classifiers.get_cross_val_scores()
            all_classifiers.save_cross_val_results(output_path + "/subject_" + str(i + 1) + "_complete.csv")

        print("Subject " + str(i+1) + " classified in " + str(time.time() - start_time) + " seconds.\n")


def transform_results(folder_path, pure_windows):
    dataframe = pd.DataFrame()

    # Read the csv
    if pure_windows:
        for i, f in enumerate(os.listdir(folder_path)):
            if f.endswith('.csv'):
                df = pd.read_csv(folder_path + '/' + f)[
                    ['classifier', 'precision_0', 'f1_0', 'precision_1', 'f1_1', 'accuracy']]
                df.insert(0, 'subject', i + 1)
                dataframe = pd.concat([dataframe, df], ignore_index=True)
    else:
        for i, f in enumerate(os.listdir(folder_path)):
            if f.endswith('.csv'):
                df = pd.read_csv(folder_path + '/' + f)[['classifier', 'mean_accuracy', 'std_accuracy']]
                df.insert(0, 'subject', i + 1)
                dataframe = pd.concat([dataframe, df], ignore_index=True)

    # Pivot table
    df_pivot = dataframe.pivot(index='subject', columns='classifier')

    # Flatten multi-index columns and reset the index
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]
    df_pivot.reset_index(inplace=True)

    # Export to csv to the same input folder
    df_pivot.to_csv(folder_path + '/full_results.csv', index=False)


def run_all(individual, win_size, pure_windows, dataset_input_path, n_subject=None):
    if not individual and n_subject is not None:
        raise Exception("If 'individual' is False, then a single subject cannot be selected")
    # Get the output paths
    hyperparams_output_path = get_routes('hyperparams', individual, win_size, pure_windows)
    result_output_path = get_routes('results', individual, win_size, pure_windows)

    # Load the dataset
    dataset = get_dataset(dataset_input_path, individual, win_size, pure_windows)

    # Optimize hyperparameters
    if n_subject is not None:  # Then individual must be False
        hyperparams_output_path += f"/subject_{n_subject}.csv"
        get_optimized_hyperparameters_for_one_subject(dataset, hyperparams_output_path, n_subject)
    else:
        hyperparams_output_path += ".csv"
        if individual:
            get_optimized_hyperparameters_for_all_subjects(dataset, hyperparams_output_path)
        else:
            get_optimized_hyperparameters_for_all_subjects_one_optimization(dataset, hyperparams_output_path)

    # Get accuracy results
    if individual:
        get_estimated_performance(dataset, hyperparams_output_path, result_output_path, pure_windows, n_subject)
    else:
        get_estimated_performance_one_optimization(dataset, hyperparams_output_path, result_output_path, pure_windows)

    transform_results(result_output_path, pure_windows)


"""
ORGANIZACIÓN DE LAS RUTAS
out/results|hyperparams|datasets
    individual/
        10s/
            pure_windows
            all_windows
        8s/
            pure_windows
            all_windows
        5s/
            pure_windows
            all_windows
    collective/
        10s/
            pure_windows
            all_windows
        8s/
            pure_windows
            all_windows
        5s/
            pure_windows
            all_windows
"""
