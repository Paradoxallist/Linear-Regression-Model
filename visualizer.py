import matplotlib.pyplot as plt
import numpy as np
from config import TRUE_WEIGHTS, TRUE_BIAS

# Plot predicted vs actual values on test set
def plot_real_vs_predicted(y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_true)), y_true, label="Actual Price", color="blue")
    plt.plot(range(len(y_pred)), y_pred, label="Predicted Price", color="orange")
    plt.title("Actual vs Predicted Prices (Final Epoch)")
    plt.xlabel("Sample Index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot loss (MSE) over training epochs
def plot_loss_over_epochs(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(losses)), losses, color="red")
    plt.title("Loss (MSE) Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Show how the model's prediction for one sample evolved over time
def plot_prediction_convergence(epoch_predictions, y_true, index=0):
    predicted_values = [epoch[index] for epoch in epoch_predictions]
    true_value = y_true.iloc[index]
    plt.figure(figsize=(10, 5))
    plt.plot(predicted_values, label="Predicted", color="green")
    plt.axhline(y=true_value, color="black", linestyle="--", label="Actual")
    plt.title(f"Prediction Convergence for Sample #{index}")
    plt.xlabel("Epoch")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot how each weight (denormalized) evolved over training
def plot_weights_evolution(epoch_weights, mean, std):
    true_weights = TRUE_WEIGHTS + [TRUE_BIAS]  # Known ground truth
    labels = ["Area", "Location Quality", "Renovation Quality", "Bias"]
    colors = ["blue", "green", "orange", "purple"]

    epochs = range(len(epoch_weights))

    # Extract weights and bias over time, and denormalize them
    weight_0 = [w[0][0] / std[0] for w in epoch_weights]
    weight_1 = [w[0][1] / std[1] for w in epoch_weights]
    weight_2 = [w[0][2] / std[2] for w in epoch_weights]
    biases   = [w[1] - np.sum((mean / std) * w[0]) for w in epoch_weights]

    values_over_time = [weight_0, weight_1, weight_2, biases]

    plt.figure(figsize=(10, 6))
    for i, (weight_values, true_val, label, color) in enumerate(zip(values_over_time, true_weights, labels, colors)):
        plt.plot(epochs, weight_values, label=f"{label} (model)", color=color)
        plt.axhline(y=true_val, linestyle="--", color=color, label=f"{label} (true)")

    plt.title("Evolution of Denormalized Weights Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Coefficient Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
