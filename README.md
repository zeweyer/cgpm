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

````

## Data File
This code requires a data file named `analysis/new_force_pairs.csv`, containing force and distance data. Make sure this data file is available at the specified path, or modify the path in the code accordingly.
The data file should contain the following columns:
-`r: Distance data
-`F: Force data

## Example Usage
1. Data Preparation
Ensure the data file `force_pairs.csv` is available and the path is correct.
2. Running the Code
Run the following command to start training the model:
```bash
python symbolic_regression_model.py
```
3. Using the `cgpm` Class
You can also directly use the cgpm class in a Python script. Here's an example:
```bash
from symbolic_regression import cgpm

# Create an instance of cgpm with the data path
model = cgpm(data_path)

# Start the training process by calling the instance
model()  # This triggers symbolic regression training
```

## Result
Once the training is complete, the results will be saved to a file named `gp_comparison_summary.csv`. This file contains performance metrics for each model, including training and validation errors.

## Notes
1. Make sure the data file path is correct. The `data_path` should point to the `force_pairs.csv` file.
2. You can adjust hyperparameters such as `population_size` and `generations` in the cgpm class to optimize the symbolic regression process.
3. The code will generate multiple `.pkl` model files during execution. These models can be saved or loaded as needed.

## Contribution
If you have any suggestions or issues, feel free to submit an Issue or submit a Pull Request.

## Author
Ziwei Yin
Contact: zewey.foster@gmail.com
