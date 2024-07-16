from src.application.adapters.persistence.repository import get_subject_loc, get_windowed_subject_loc, \
    get_optimized_general_loc, get_results_loc, is_optimized_subject_for_algorithm, create_if_not_exists_dir, \
    get_optimized_for_subject_and_algorithm_loc, is_windowed, get_exact_window_index_loc
from src.application.core.feature_extraction.data_loader import DataLoader
from src.application.core.state_classification.classifiers import CustomizedClassifier
from src.application.core.state_classification.hyperparameter_optimization import SingleOptimizer


def make_windows(subject: str, window: int):
    subject_loc = get_subject_loc(subject)
    out_window_loc = get_windowed_subject_loc(subject, window)
    exact_index_loc = get_exact_window_index_loc(subject, window)
    return DataLoader(subject_loc, out_window_loc, window, exact_index_loc).dataset


def optimize_for_algorithm(subject: str, algorithm: str, window: int):
    if is_optimized_subject_for_algorithm(subject, algorithm, window):
        return "Subject already optimized for the given algorithm and time window."

    subject_loc = get_subject_loc(subject)
    data_loc = get_windowed_subject_loc(subject, window)
    exact_index_loc = get_exact_window_index_loc(subject, window)
    general_opt_loc = get_optimized_general_loc(algorithm, window)
    hyperparams_output_loc = get_optimized_for_subject_and_algorithm_loc(subject, algorithm, window)

    create_if_not_exists_dir(hyperparams_output_loc)

    # window the data and save it in data_loc
    data = DataLoader(subject_loc, data_loc, window, exact_index_loc).dataset

    # optimize the algorithm
    SingleOptimizer(data, algorithm, general_opt_loc, hyperparams_output_loc)

    return "Subject optimized successfully."


def train_algorithm(subject: str, algorithm: str, window: int, train_set_size: int, use_optimized_hyperparams: bool):

    subject_loc = get_subject_loc(subject)
    data_loc = get_windowed_subject_loc(subject, window)
    exact_index_loc = get_exact_window_index_loc(subject, window)

    if use_optimized_hyperparams:
        if not is_optimized_subject_for_algorithm(subject, algorithm, window):
            return "The subject has not been optimized for the given algorithm and time window. Please optimize first."
        hyperparams_loc = get_optimized_for_subject_and_algorithm_loc(subject, algorithm, window)
    else:
        hyperparams_loc = get_optimized_general_loc(algorithm, window)

    output_loc = get_results_loc(subject, algorithm, use_optimized_hyperparams, window)

    if not is_windowed(subject, window):
        make_windows(subject, window)

    # Call classifier
    cl = CustomizedClassifier(data_loc, hyperparams_loc, output_loc, exact_index_loc, train_set_size)
    return cl.results


