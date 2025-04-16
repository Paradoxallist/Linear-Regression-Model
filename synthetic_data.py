import pandas as pd
import numpy as np
from config import (
    N_SAMPLES, NOISE_STD, TRUE_WEIGHTS, TRUE_BIAS, SEED,
    AREA_RANGE, LOCATION_QUALITY_RANGE, RENOVATION_QUALITY_RANGE
)

class SyntheticDataGenerator:
    def __init__(self, n_samples=N_SAMPLES, noise_std=NOISE_STD):
        self.n_samples = n_samples
        self.noise_std = noise_std

    def generate(self):
        # Seed for reproducibility
        np.random.seed(SEED)

        # Feature generation
        area = np.random.uniform(*AREA_RANGE, size=self.n_samples)
        location_quality = np.random.randint(*LOCATION_QUALITY_RANGE, size=self.n_samples)
        renovation_quality = np.random.randint(*RENOVATION_QUALITY_RANGE, size=self.n_samples)

        # Random noise
        noise = np.random.normal(0, self.noise_std, size=self.n_samples)

        # Target variable calculated by the true formula
        price = (
                TRUE_WEIGHTS[0] * area +
                TRUE_WEIGHTS[1] * location_quality +
                TRUE_WEIGHTS[2] * renovation_quality +
                TRUE_BIAS +
                noise
        )

        # Create pandas DataFrame with English column names
        df = pd.DataFrame({
            'Area': area,
            'Location Quality': location_quality,
            'Renovation Quality': renovation_quality,
            'Price': price
        })
        return df
