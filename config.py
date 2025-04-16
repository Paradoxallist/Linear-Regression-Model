# === Dataset Configuration ===
N_SAMPLES = 10000          # Number of synthetic data points
NOISE_STD = 5.0            # Standard deviation of Gaussian noise

# === Data Generation Formula (Ground Truth) ===
TRUE_WEIGHTS = [40, 8, 2]  # Coefficients for Area, Location Quality, Renovation Quality
TRUE_BIAS = 10             # Bias term

# === Training Configuration ===
LEARNING_RATE = 0.0005     # Learning rate for gradient descent
EPOCHS = 20000             # Number of training epochs

# === Test Prediction Example ===
EXAMPLE_INPUT = {
    "area": 70,
    "location_quality": 9,
    "renovation_quality": 4
}

# === Random Seed ===
SEED = 0

# === Feature Generation Ranges ===
AREA_RANGE = (30, 100)                  # Square meters (float)
LOCATION_QUALITY_RANGE = (1, 11)        # Integer range: 1 to 10 inclusive
RENOVATION_QUALITY_RANGE = (1, 6)       # Integer range: 1 to 5 inclusive
