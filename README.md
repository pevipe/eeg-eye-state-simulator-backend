
# Requirements
## hyperopt-sklearn
To install `hyperopt-sklearn`, the following command must be manually run:
```bash
pip install git+https://github.com/hyperopt/hyperopt-sklearn
```
Although it could be automatically installed with the `requirements.txt` file, 
the version of the package (0.1.0) would not be compatible with versions of Python
3.11. Newer versions of the library, such as 0.1.3, are needed.

## Configuration
For keeping persistence when running the application with the frontend, the environment
variable `SIMULATOR_PERSISTENT_PATH` must be set to the path where the data of the application
must be stored. As a starting point, a `zip` file will be provided, containing the data and 
optimized hyperparameters for the seven starting subjects. This must be decomposed inside the
indicated directory.

# Run the API
To run the backend for the eeg-simulator, using pipenv (`pipenv install`) is recommended:
```bash
pipenv run uvicorn src.main:create_app --factory --reload
```

