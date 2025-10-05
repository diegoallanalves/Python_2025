import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Generate Dummy Data
# Generating a dataset with 100 points
np.random.seed(42)  # For reproducibility
X = np.linspace(1, 100, 100).reshape(-1, 1)  # Features (1D for chart simplicity)
y = X * 2 + np.random.normal(0, 10, size=X.shape)  # Target variable with noise

# 2. Split Data into Train and Test
# Using a 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Plot the Train-Test Split
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color="blue", label="Train Data", alpha=0.7)

# Plot testing data
plt.scatter(X_test, y_test, color="red", label="Test Data", alpha=0.7)

# Add chart details
plt.title("Train/Test Split Visualization")
plt.xlabel("Feature (X)")
plt.ylabel("Target (y)")
plt.axvline(X_train.max(), color="green", linestyle="--", label="Train-Test Split Line")
plt.legend()
plt.grid()

# Show the plot
plt.show()

# 4. Print the split details
print(f"Total data points: {len(X)}")
print(f"Training data points: {len(X_train)}")
print(f"Testing data points: {len(X_test)}")
