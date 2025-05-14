import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic water quality data
np.random.seed(42)
n_samples = 1000
ph = np.random.uniform(0, 14, n_samples)
turbidity = np.random.uniform(0, 100, n_samples)
temperature = np.random.uniform(0, 100, n_samples)
conductivity = np.random.uniform(0, 2000, n_samples)
X = np.column_stack((ph, turbidity, temperature, conductivity))

# Simulate labels: potable (1) if parameters are in safe ranges, else contaminated (0)
y = np.where(
    (6.5 <= ph) & (ph <= 8.5) &
    (turbidity <= 5) &
    (temperature <= 30) &
    (conductivity <= 1000), 1, 0
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, "water_quality_model.pkl")
print("Model saved as water_quality_model.pkl")