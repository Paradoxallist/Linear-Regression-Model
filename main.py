from synthetic_data import SyntheticDataGenerator
from model import LinearRegressionManual
from visualizer import plot_real_vs_predicted, plot_loss_over_epochs, plot_prediction_convergence, plot_weights_evolution
from sklearn.model_selection import train_test_split

def main():
    # Generate synthetic dataset with 10,000 samples and Gaussian noise
    data = SyntheticDataGenerator(n_samples=10000, noise_std=5).generate()

    # Split into features (X) and target (y)
    X = data.drop(columns=["Price"])
    y = data["Price"]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize linear regression model with gradient descent
    model = LinearRegressionManual(learning_rate=0.0005, epochs=20000)
    model.fit(X_train, y_train)

    # Recover learned weights in original scale (denormalization)
    denorm_w, denorm_b = model.denormalize_weights()

    # Print ground truth formula used in data generation
    print("\nTrue formula:")
    print("Price = 40 * Area + 8 * Location Quality + 2 * Renovation Quality + 10")

    # Print model's learned approximation of the formula
    print("\nLearned formula (denormalized):")
    print(f"Price = {denorm_w[0]:.4f} * Area + {denorm_w[1]:.4f} * Location Quality + {denorm_w[2]:.4f} * Renovation Quality + {denorm_b:.4f}")

    # Visualization: compare predicted and actual prices on test set
    plot_real_vs_predicted(y_test, model.predict(X_test))

    # Visualization: loss function trend over training epochs
    plot_loss_over_epochs(model.losses)

    # Visualization: prediction for a specific object across training
    plot_prediction_convergence(model.epoch_predictions, y_train, index=0)

    # Visualization: weight (coefficient) convergence over epochs
    plot_weights_evolution(model.epoch_weights, model.mean, model.std)

    # Example of manual prediction using trained model
    print("\nManual model prediction examples:")
    area, location_quality, renovation_quality = 70, 9, 4
    predicted = model.calculate(area, location_quality, renovation_quality)

    # Compute true price using original formula
    expected = 40 * area + 8 * location_quality + 2 * renovation_quality + 10

    # Compare model prediction to true value
    print(f"Predicted: {predicted:.2f}")
    print(f"Expected:  {expected:.2f}")

if __name__ == "__main__":
    main()
