import numpy as np

class LinearRegressionManual:
    def __init__(self, learning_rate=0.0005, epochs=200):
        # Training parameters
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters
        self.weights = None
        self.bias = 0

        # Tracking for diagnostics and visualization
        self.losses = []
        self.epoch_predictions = []
        self.epoch_weights = []

        # Statistics for normalization
        self.mean = None
        self.std = None

    def normalize(self, X):
        # Z-score normalization of features
        return (X - self.mean) / self.std

    def denormalize_weights(self):
        # Convert normalized weights back to original scale
        denorm_weights = self.weights / self.std
        denorm_bias = self.bias - np.sum((self.mean / self.std) * self.weights)
        return denorm_weights, denorm_bias

    def fit(self, X, y):
        # Convert pandas to numpy if necessary
        X = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        # Compute normalization statistics from training data
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        X = self.normalize(X)

        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent training loop
        for epoch in range(self.epochs):
            y_pred = self._predict_internal(X)
            error = y_pred - y

            self.epoch_predictions.append(y_pred.copy())
            self.losses.append(np.mean(error ** 2))
            self.epoch_weights.append((self.weights.copy(), self.bias))

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Normalize input and return predictions
        X = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        X = self.normalize(X)
        return self._predict_internal(X)

    def _predict_internal(self, X):
        # Core linear model: y = Wx + b
        return np.dot(X, self.weights) + self.bias

    def calculate(self, area, location_quality, renovation_quality):
        # Manual prediction for raw (unnormalized) input
        features = np.array([area, location_quality, renovation_quality])
        features = (features - self.mean) / self.std
        return np.dot(self.weights, features) + self.bias
