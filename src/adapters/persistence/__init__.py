import os

base_directory = os.environ.get('SIMULATOR_PERSISTENT_PATH')

for folder in ['datasets', 'hyperparams', 'results']:
    path = os.path.join(base_directory, folder)
    if not os.path.exists(path):
        os.makedirs(path)
