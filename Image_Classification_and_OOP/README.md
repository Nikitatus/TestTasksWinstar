# Image Classification + OOP

In this task three classification models are implemented on the public MNIST dataset using an OOP approach.

## Models
The repository contains 3 classification models:
1. **Random Forest (`rf`)**:
2. **Feed-Forward Neural Network (`nn`)**:
3. **Convolutional Neural Network (`cnn`)**:

Each model is a separate class (`RandomForestModel`, `FeedForwardModel`, `CNNModel`) that implements the `MnistClassifierInterface` with two abstract methods:
- `train(X, y)`
- `predict(X)`

All models are accessible through a unified `MnistClassifier` wrapper class.

## Project Structure
- `interface.py`: Defines the `MnistClassifierInterface`.
- `models.py`: Implementations of Random Forest, Feed-Forward NN, and CNN.
- `classifier.py`: The `MnistClassifier` wrapper class.
- `helper_models.py`: PyTorch model definitions for NN and CNN.
- `utils.py`: Utility functions.
- `demo.ipynb`: Demonstration notebook.

## Setup
### Requirements
All required packages are listed in `requirements.txt`:
Requirements can be installed via pip:
```bash
pip install -r requirements.txt
```

## Usage
Example of usage

```python
from classifier import MnistClassifier

# Initialize models
forest = MnistClassifier(algorithm="rf", n_estimators=100)
feed_forward = MnistClassifier(algorithm="nn", lr=1e-3, epochs=10)
convolutional = MnistClassifier(algorithm="cnn", lr=1e-3, epochs=10)

# Train
forest.train(X_train, y_train)

# Predict
predictions = forest.predict(X_test)
```