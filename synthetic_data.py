import pandas as pd
import numpy as np

class SyntheticDataGenerator:
    def __init__(self, n_samples=100, noise_std=5.0):
        self.n_samples = n_samples
        self.noise_std = noise_std

    def generate(self):
        # Seed for reproducibility
        np.random.seed(0)

        # Feature generation
        area = np.random.uniform(30, 100, size=self.n_samples)                # Apartment area in square meters
        location_quality = np.random.randint(1, 11, size=self.n_samples)       # Quality of location (1–10)
        renovation_quality = np.random.randint(1, 6, size=self.n_samples)      # Renovation quality (1–5)

        # True bias and random noise
        bias = 10
        noise = np.random.normal(0, self.noise_std, size=self.n_samples)

        # Target variable calculated by the true formula
        price = 40 * area + 8 * location_quality + 2 * renovation_quality + bias + noise

        # Create pandas DataFrame with English column names
        df = pd.DataFrame({
            'Area': area,
            'Location Quality': location_quality,
            'Renovation Quality': renovation_quality,
            'Price': price
        })
        return df
