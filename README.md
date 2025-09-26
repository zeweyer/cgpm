# CoarseGrainedPotentialModel-SR(cgpm)
This project applies symbolic regression techniques to model coarse-grained potentials from molecular dynamics data. The goal is to automate the extraction of interpretable, physically-based force and energy functions for molecular systems.

## Installation
pip install gplearn
````
pip install gplearn
````
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
