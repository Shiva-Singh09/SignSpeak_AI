
import numpy as np


# ==========================================
# STEP 1: UNDERSTANDING A SINGLE NEURON
# ==========================================

# Input features
inputs = np.array([0.5, 0.8, 0.2])

# Weights assigned to each input
weights = np.array([0.4, 0.6, 0.3])

# Bias
bias = 0.2


# ==========================================
# ACTIVATION FUNCTIONS
# ==========================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


# ==========================================
# NEURON FORWARD CALCULATION
# ==========================================

# Weighted sum
z = np.dot(inputs, weights) + bias

print("=" * 45)
print("SINGLE NEURON - FORWARD CALCULATION")
print("=" * 45)

print("\nInputs:", inputs)
print("Weights:", weights)
print("Bias:", bias)

print("\nWeighted sum (z):", z)

print("\nActivation outputs:")
print("Sigmoid:", sigmoid(z))
print("ReLU:", relu(z))
print("Tanh:", tanh(z))


# ==========================================
# UNDERSTANDING WEIGHT CONTRIBUTION
# ==========================================

print("\nIndividual weighted contributions:")

for i in range(len(inputs)):
    contribution = inputs[i] * weights[i]
    print(
        f"Input {i + 1}: {inputs[i]} x "
        f"{weights[i]} = {contribution}"
    )

print("\nFinal weighted sum:", z)
print("Bias added:", bias)
