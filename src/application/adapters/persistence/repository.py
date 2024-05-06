import os

base_directory = os.environ.get('SIMULATOR_PERSISTENT_PATH')


def get_loaded_subjects():
    all_in_datasets = os.listdir(os.path.join(base_directory, 'datasets'))
    # Suppose all folders are loaded subjects
    return [subject for subject in all_in_datasets if os.path.isdir(os.path.join(base_directory, 'datasets', subject))]


def is_loaded_subject(subject):
    return os.path.exists(os.path.join(base_directory, 'datasets', subject, f'{subject}.csv'))


def get_subject_loc(subject):
    return os.path.join(base_directory, 'datasets', subject, f'{subject}.csv')


def upload_subject(subject_name, content):
    loc = get_subject_loc(subject_name)

    if is_loaded_subject(subject_name):
        with open(loc, 'r') as f:
            file_content = f.read(content)
            if file_content == content:
                return "The file was already uploaded."
        with open(loc, 'w') as f:  # Overwrite the file
            f.write(content)
            return "Successfully uploaded file."


def is_windowed(subject, window):
    return os.path.exists(os.path.join(base_directory, 'datasets', subject, f"{window}s.csv"))


def get_windowed_subject_loc(subject, window):
    return os.path.join(base_directory, 'datasets', subject, f"{window}s.csv")


def is_optimized_subject_for_algorithm(subject, algorithm, window=10):
    return os.path.exists(os.path.join(base_directory, 'hyperparams', subject, f"{window}s", f"{algorithm}.csv"))


def get_optimized_for_subject_and_algorithm_loc(subject, algorithm, window=10):
    return os.path.join(base_directory, 'hyperparams', subject, f"{window}s", f"{algorithm}.csv")


def get_optimized_general_loc(algorithm, window=10):
    return os.path.join(base_directory, 'hyperparams_general', f'{window}s', f"{algorithm}.csv")


def has_results(subject, algorithm, window=10):
    return os.path.exists(os.path.join(base_directory, 'results', subject, f"{window}s", algorithm))


def get_results_loc(subject, algorithm, window=10):
    return os.path.join(base_directory, 'results', subject, f"{window}s", algorithm)


def init_subject_routes(subject, window):
    for folder in ['datasets', 'hyperparams', 'results']:
        path = os.path.join(base_directory, folder, subject, f"{window}s")
        if not os.path.exists(path):
            os.makedirs(path)
