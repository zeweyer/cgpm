# CoarseGrainedPotentialModel-SR(cgpm)
This project applies symbolic regression techniques to model coarse-grained potentials from molecular dynamics data. The goal is to automate the extraction of interpretable, physically-based force and energy functions for molecular systems.

## Dependencies

Before running the code, make sure you have the following dependencies installed:

- `pandas`
- `gplearn`
- `scikit-learn`
- `joblib`
- `numpy`

You can install all dependencies with the following command:
### Using `pip` to Install:

1. Clone or download the project:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
### `requirements.txt` Example:
To make it easier for others to install dependencies, include the following in your `requirements.txt` file:

```txt
pandas
gplearn
scikit-learn
joblib
numpy

## Example Usage
````
from symbolic_regression import cgpm

# Load your data
data = load_data('data/force_pairs.csv')

# Train the model
model = cgpm()
model.train(data)

# Evaluate the model
model.evaluate()

````

## Result
The symbolic regression model successfully extracts a physically interpretable potential function that can be used in coarse-grained molecular simulations. For detailed results, check the 'results/' folder.
