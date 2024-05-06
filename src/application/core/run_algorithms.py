from src.application.adapters.persistence.repository import get_subject_loc, is_windowed, get_windowed_subject_loc, \
    get_optimized_general_loc, get_results_loc, is_optimized_subject_for_algorithm
from src.application.core.feature_extraction.data_loaders import SingleDatasetLoader
from src.application.core.state_classification.hyperparameter_optimization import SingleOptimizer


def optimize_for_algorithm(subject: str, algorithm: str, window: int):
    if is_optimized_subject_for_algorithm(subject, algorithm, window):
        return "Subject already optimized for the given algorithm and time window."

    subject_loc = get_subject_loc(subject)
    data_loc = get_windowed_subject_loc(subject, window)
    general_opt_loc = get_optimized_general_loc(algorithm, window)
    output_path = get_results_loc(subject, algorithm, window)

    # window the data and save it in data_loc
    data = SingleDatasetLoader(subject_loc, data_loc, window).dataset

    # optimize the algorithm
    SingleOptimizer(data, algorithm, general_opt_loc, output_path)

    return "Subject optimized successfully."


def train_algorithm(subject: str, algorithm: str, window: int, train_set_size: int, use_optimized_hyperparams: bool):
    pass
