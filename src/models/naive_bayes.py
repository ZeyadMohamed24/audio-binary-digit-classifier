import numpy as np

class Gaussian_Naive_Bayes:
    def __init__(self):
        self.__classes = {}
        self.__means = {}
        self.__vars = {}
        self.__priors = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.__classes = np.unique(y)
        for Class in self.__classes:
            X_class = X[y == Class]
            self.__means[Class] = np.mean(X_class, axis=0)
            self.__vars[Class] = np.var(X_class, axis=0)
            self.__priors[Class] = X_class.shape[0] / X.shape[0]

    def gaussian(self, X: np.ndarray, Class) -> float:
        mean = self.__means[Class]
        var = self.__vars[Class]
        numerator = np.exp(-((X - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            posteriors = {}
            for Class in self.__classes:
                prior = self.__priors[Class]
                likelihoods = self.gaussian(x, Class)
                class_conditional = np.prod(likelihoods)
                posteriors[Class] = prior * class_conditional
            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)
        return np.array(predictions)
