from src.application.adapters.persistence.repository import get_subject_loc, get_windowed_subject_loc, \
    get_optimized_general_loc, get_results_loc, is_optimized_subject_for_algorithm, create_if_not_exists_dir, \
    get_optimized_for_subject_and_algorithm_loc
from src.application.core.feature_extraction.data_loaders import SingleDatasetLoader
from src.application.core.state_classification.hyperparameter_optimization import SingleOptimizer


def make_windows(subject: str, window: int):
    subject_loc = get_subject_loc(subject)
    out_window_loc = get_windowed_subject_loc(subject, window)
    SingleDatasetLoader(subject_loc, out_window_loc, window)
    return SingleDatasetLoader(subject_loc, out_window_loc, window).dataset


def optimize_for_algorithm(subject: str, algorithm: str, window: int):
    if is_optimized_subject_for_algorithm(subject, algorithm, window):
        print("Subject already optimized for the given algorithm and time window.")
        return "Subject already optimized for the given algorithm and time window."

    subject_loc = get_subject_loc(subject)
    data_loc = get_windowed_subject_loc(subject, window)
    general_opt_loc = get_optimized_general_loc(algorithm, window)
    hyperparams_output_loc = get_optimized_for_subject_and_algorithm_loc(subject, algorithm, window)

    create_if_not_exists_dir(hyperparams_output_loc)

    # window the data and save it in data_loc
    data = SingleDatasetLoader(subject_loc, data_loc, window).dataset

    # optimize the algorithm
    SingleOptimizer(data, algorithm, general_opt_loc, hyperparams_output_loc)

    return "Subject optimized successfully."


def train_algorithm(subject: str, algorithm: str, window: int, train_set_size: int, use_optimized_hyperparams: bool):
    pass
