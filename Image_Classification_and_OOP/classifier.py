from interface import MnistClassifierInterface
from models import CNNModel, RandomForestModel, FeedForwardModel

class MnistClassifier(MnistClassifierInterface):
    algorithms = {"cnn": CNNModel, "rf": RandomForestModel, "nn": FeedForwardModel}

    def __init__(self, algorithm, **kwargs):
        if algorithm not in self.algorithms:
            raise ValueError(f"The algorithm {algorithm} is not supported, dude!")

        self.algorithm_name = algorithm
        self.classifier = self.algorithms[self.algorithm_name](**kwargs)

    def train(self, X, y):
        return self.classifier.train(X, y)

    def predict(self, X):
        return self.classifier.predict(X)