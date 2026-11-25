
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(precision=8, suppress=True)


# ==========================================
# EXPERIMENT CONFIGURATION
# ==========================================

LAYERS = 20
NEURONS = 32

rng = np.random.default_rng(42)


# ==========================================
# ACTIVATION FUNCTIONS AND DERIVATIVES
# ==========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


# ==========================================
# ORTHOGONAL WEIGHT INITIALIZATION
# ==========================================

def create_orthogonal_weights():

    weights = []

    for _ in range(LAYERS):

        matrix = rng.normal(
            0, 1, (NEURONS, NEURONS)
        )

        q, r = np.linalg.qr(matrix)

        # Correct signs for consistent QR output
        signs = np.sign(np.diag(r))
        signs[signs == 0] = 1

        q = q * signs[np.newaxis, :]

        weights.append(q)

    return weights


base_weights = create_orthogonal_weights()


# ==========================================
# FORWARD AND BACKWARD EXPERIMENT
# ==========================================

def run_experiment(name, activation, derivative,
                  input_scale, weight_scale=1.0):

    x = rng.normal(
        0, input_scale, (1, NEURONS)
    )

    weights = [
        w * weight_scale for w in base_weights
    ]

    activations = [x]
    z_values = []

    # FORWARD PASS
    for w in weights:

        z = activations[-1] @ w
        a = activation(z)

        z_values.append(z)
        activations.append(a)

    # BACKWARD PASS
    grad = np.ones((1, NEURONS)) / np.sqrt(NEURONS)

    gradient_norms = []
    mean_derivatives = []

    for layer in reversed(range(LAYERS)):

        local_derivative = derivative(
            z_values[layer]
        )

        delta = grad * local_derivative

        gradient_norms.append(
            float(np.linalg.norm(delta))
        )

        mean_derivatives.append(
            float(np.mean(local_derivative))
        )

        grad = delta @ weights[layer].T

    gradient_norms.reverse()
    mean_derivatives.reverse()

    return {
        "name": name,
        "gradients": gradient_norms,
        "derivatives": mean_derivatives
    }


# ==========================================
# RUN FOUR EXPERIMENTS
# ==========================================

results = []

results.append(
    run_experiment(
        "Sigmoid",
        sigmoid,
        sigmoid_derivative,
        input_scale=0.1
    )
)

results.append(
    run_experiment(
        "Tanh - Small Input",
        tanh,
        tanh_derivative,
        input_scale=0.1
    )
)

results.append(
    run_experiment(
        "Tanh - Large Input",
        tanh,
        tanh_derivative,
        input_scale=5.0
    )
)

results.append(
    run_experiment(
        "ReLU",
        relu,
        relu_derivative,
        input_scale=0.1,
        weight_scale=np.sqrt(2)
    )
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("=" * 75)
print("DEEP NETWORK - GRADIENT ANALYSIS")
print("=" * 75)

print("\nNetwork configuration:")
print("Hidden layers:", LAYERS)
print("Neurons per layer:", NEURONS)
print("Initialization: Orthogonal weights")

for result in results:

    print("\n" + "-" * 75)
    print(result["name"])
    print("-" * 75)

    print(
        f"{'Layer':<10}"
        f"{'Gradient Norm':<25}"
        f"{'Mean Activation Derivative':<30}"
    )

    for i in range(LAYERS):

        print(
            f"{i + 1:<10}"
            f"{result['gradients'][i]:<25.10e}"
            f"{result['derivatives'][i]:.8f}"
        )

    print(
        "\nFinal gradient norm:",
        result["gradients"][0]
    )

    print(
        "Final-layer gradient norm:",
        result["gradients"][-1]
    )


# ==========================================
# SAVE GRADIENT COMPARISON GRAPH
# ==========================================

output_dir = (
    Path(__file__).resolve().parents[3]
    / "outputs"
    / "experiments"
    / "backprop_vanishing"
)

output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(11, 6))

for result in results:

    values = np.maximum(
        result["gradients"], 1e-20
    )

    plt.semilogy(
        range(1, LAYERS + 1),
        values,
        marker="o",
        label=result["name"]
    )

plt.xlabel("Layer Number (Input to Output)")
plt.ylabel("Gradient Norm (Log Scale)")
plt.title("Gradient Flow Across 20 Hidden Layers")
plt.xticks(range(1, LAYERS + 1))
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

graph_path = (
    output_dir / "deep_network_gradient_comparison.png"
)

plt.savefig(graph_path, dpi=150)
plt.close()

print("\n" + "=" * 75)
print("EXPERIMENT COMPLETED")
print("=" * 75)

print("\nGraph saved at:")
print(graph_path)

