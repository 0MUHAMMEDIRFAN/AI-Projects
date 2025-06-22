import pickle

class SimpleLinearModel:
    def __init__(self):
        self.coeff = None
        self.bias = 0

    def train(self, X, y):
        # Simple linear regression (1D)
        n = len(X)
        x_mean = sum(X) / n
        y_mean = sum(y) / n

        num = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        den = sum((X[i] - x_mean) ** 2 for i in range(n))
        self.coeff = num / den
        self.bias = y_mean - self.coeff * x_mean

    def predict(self, x):
        return self.coeff * x + self.bias

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
