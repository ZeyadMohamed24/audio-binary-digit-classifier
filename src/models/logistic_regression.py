import numpy as np

class LogisticRegressionGD:
    def __init__(self, alpha: float = 0.01, iters: int = 1000, batch_size: int = None):
        self.alpha = alpha
        self.iters = iters
        self.batch_size = batch_size
        self.__w = None
        self.__b = None
        self.__losses = []

    def __compute_loss(self, y: np.ndarray, h: np.ndarray) -> float:
        m = len(y)
        loss = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return loss

    def __sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    def __batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]
        for iter in range(self.iters):
            z = np.dot(X, self.__w) + self.__b
            h = self.__sigmoid(z)
            loss = self.__compute_loss(y, h)
            self.__losses.append(loss)
            dw = (1 / m) * np.dot(X.T, (h - y))
            db = (1 / m) * np.sum(h - y)
            self.__w -= self.alpha * dw
            self.__b -= self.alpha * db

    def __stochastic_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]
        for iter in range(self.iters):
            total_loss = 0
            for i in range(m):
                z = np.dot(X[i], self.__w) + self.__b
                h = self.__sigmoid(z)
                loss = self.__compute_loss(np.array([y[i]]), np.array([h]))
                total_loss += loss
                dw = X[i] * (h - y[i])
                db = h - y[i]
                self.__w -= self.alpha * dw
                self.__b -= self.alpha * db
            self.__losses.append(total_loss / m)

    def __mini_batch_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        m = X.shape[0]
        for iter in range(self.iters):
            indices = np.random.permutation(m)
            X_rand = X[indices]
            y_rand = y[indices]
            total_loss = 0
            for i in range(0, m, self.batch_size):
                batch_X = X_rand[i : i + self.batch_size]
                batch_y = y_rand[i : i + self.batch_size]
                z = np.dot(batch_X, self.__w) + self.__b
                h = self.__sigmoid(z)
                loss = self.__compute_loss(batch_y, h)
                total_loss += loss
                dw = (1 / self.batch_size) * np.dot(batch_X.T, (h - batch_y))
                db = (1 / self.batch_size) * np.sum(h - batch_y)
                self.__w -= self.alpha * dw
                self.__b -= self.alpha * db
            self.__losses.append(total_loss / self.batch_size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[1]
        self.__w = np.random.randn(n) * 0.01
        self.__b = 0
        if self.batch_size is None:
            self.__batch_gradient_descent(X, y)
        elif self.batch_size == 1:
            self.__stochastic_gradient_descent(X, y)
        else:
            self.__mini_batch_gradient_descent(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.__w) + self.__b
        h = self.__sigmoid(z)
        return (h >= 0.5).astype(int)
