from abc import ABC, abstractmethod

class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X, y):
        """
        Train the MNIST classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width)
            Training data.

        y : np.ndarray of shape (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict labels for input images.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, height, width)
            Sample images to classify.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Returns predicted values.
        """
        pass