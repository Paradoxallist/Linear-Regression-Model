from synthetic_data import SyntheticDataGenerator
from model import LinearRegressionManual
from visualizer import plot_real_vs_predicted, plot_loss_over_epochs, plot_prediction_convergence, plot_weights_evolution
from sklearn.model_selection import train_test_split
from config import LEARNING_RATE, EPOCHS, EXAMPLE_INPUT, TRUE_WEIGHTS, TRUE_BIAS

def main():
    # Generate synthetic dataset using configured parameters
    data = SyntheticDataGenerator().generate()

    # Split into features (X) and target (y)
    X = data.drop(columns=["Price"])
    y = data["Price"]

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize linear regression model with configured learning rate and epochs
    model = LinearRegressionManual(learning_rate=LEARNING_RATE, epochs=EPOCHS)
    model.fit(X_train, y_train)

    # Recover learned weights in original scale (denormalization)
    denorm_w, denorm_b = model.denormalize_weights()

    # Print ground truth formula used in data generation
    print("\nTrue formula:")
    print(f"Price = {TRUE_WEIGHTS[0]} * Area + {TRUE_WEIGHTS[1]} * Location Quality + {TRUE_WEIGHTS[2]} * Renovation Quality + {TRUE_BIAS}")

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
    print("\nManual model prediction example:")
    area = EXAMPLE_INPUT["area"]
    location_quality = EXAMPLE_INPUT["location_quality"]
    renovation_quality = EXAMPLE_INPUT["renovation_quality"]
    predicted = model.calculate(area, location_quality, renovation_quality)

    # Compute true price using original formula
    expected = (
        TRUE_WEIGHTS[0] * area +
        TRUE_WEIGHTS[1] * location_quality +
        TRUE_WEIGHTS[2] * renovation_quality +
        TRUE_BIAS
    )

    # Compare model prediction to true value
    print(f"Predicted: {predicted:.2f}")
    print(f"Expected:  {expected:.2f}")

if __name__ == "__main__":
    main()