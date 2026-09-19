"""SOM-based multi-directional ablation (SOM-MD).

Trains a Self-Organizing Map on harmful-prompt hidden states at a given layer,
then computes k unit directions from the k best neurons toward the harmless
centroid. These directions capture the refusal manifold structure that a single
difference-in-means direction misses.

Reference: "Multi-Directional Ablation" (SOM-MD) arXiv paper.
"""

import numpy as np


def train_som(
    data: np.ndarray,
    grid_shape: tuple[int, int] = (4, 4),
    n_iter: int = 1000,
    lr_init: float = 0.5,
    sigma_init: float = 2.0,
    seed: int = 42,
) -> np.ndarray:
    """Train a Kohonen SOM on data using the standard sequential algorithm.

    Args:
        data: (n_samples, dim) harmful-prompt hidden states at layer l*.
        grid_shape: (rows, cols) for the SOM lattice.
        n_iter: number of training iterations.
        lr_init: initial learning rate (decays linearly to 0).
        sigma_init: initial neighbourhood radius (in grid units), decays to 0.
        seed: RNG seed for weight initialization.

    Returns:
        weights: (rows*cols, dim) SOM weight matrix.
    """
    n_rows, n_cols = grid_shape
    n_neurons = n_rows * n_cols
    rng = np.random.default_rng(seed)

    # Initialize weights by sampling from the data (gives better coverage than random).
    indices = rng.choice(len(data), size=n_neurons, replace=len(data) < n_neurons)
    weights = data[indices].astype(np.float64).copy()

    for it in range(n_iter):
        alpha = lr_init * (1.0 - it / n_iter)
        sigma = sigma_init * (1.0 - it / n_iter)

        # Pick a random sample and find its best matching unit.
        idx = rng.integers(len(data))
        sample = data[idx]
        dists = np.linalg.norm(weights - sample, axis=1)
        bmu = int(np.argmin(dists))

        # Update all neurons in the neighbourhood.
        for i in range(n_rows):
            for j in range(n_cols):
                neuron_idx = i * n_cols + j
                # Toroidal distance to BMU.
                bi, bj = divmod(bmu, n_cols)
                di = min(abs(i - bi), n_rows - abs(i - bi))
                dj = min(abs(j - bj), n_cols - abs(j - bj))
                dist_sq = di * di + dj * dj
                if dist_sq > max(sigma * sigma, 1e-8):
                    continue
                # Gaussian neighbourhood.
                h = np.exp(-dist_sq / (2.0 * max(sigma * sigma, 1e-8)))
                weights[neuron_idx] += alpha * h * (sample - weights[neuron_idx])

    return weights.astype(np.float32)


def compute_som_directions(
    harmful_states: np.ndarray,
    harmless_centroid: np.ndarray,
    grid_shape: tuple[int, int] = (4, 4),
    k: int = 7,
    seed: int = 42,
) -> list[np.ndarray]:
    """Train a SOM on harmful states and return k unit directions from the best
    neurons toward the harmless centroid.

    Args:
        harmful_states: (n_harmful, dim) hidden states at layer l*.
        harmless_centroid: (dim,) mean of harmless-prompt hidden states at l*.
        grid_shape: SOM lattice dimensions.
        k: number of directions to return (top-k neurons by activation).
        seed: RNG seed for SOM initialization.

    Returns:
        List of k unit vectors (dim,), each pointing from a SOM neuron toward
        the harmless centroid.
    """
    weights = train_som(harmful_states, grid_shape=grid_shape, seed=seed)
    n_neurons = len(weights)

    # Activation: how many harmful samples have this neuron as BMU.
    dists = np.linalg.norm(harmful_states[:, None, :] - weights[None, :, :], axis=2)
    bmu_counts = np.bincount(np.argmin(dists, axis=1), minlength=n_neurons)

    # Pick top-k neurons by activation count (most representative of harmful data).
    top_k = np.argsort(bmu_counts)[::-1][:k]

    directions = []
    for neuron_idx in top_k:
        direction = harmless_centroid - weights[neuron_idx]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            directions.append((direction / norm).astype(np.float32))
        else:
            directions.append(np.zeros_like(harmless_centroid, dtype=np.float32))

    return directions


def select_best_layer(
    harmful_states_all_layers: np.ndarray,
    harmless_states_all_layers: np.ndarray,
    grid_shape: tuple[int, int] = (4, 4),
) -> int:
    """Find the layer where ablating a single direction most reduces refusal.

    Uses the Arditi et al. (2024) refusal metric: for each layer, compute the
    difference-in-means direction and measure how much it separates harmful from
    harmless states. The layer with maximum separation is the best ablation target.

    Args:
        harmful_states_all_layers: (n_harmful, n_layers, dim)
        harmless_states_all_layers: (n_harmless, n_layers, dim)

    Returns:
        Index of the best layer (0-indexed into the hidden-state array).
    """
    n_layers = harmful_states_all_layers.shape[1]
    scores = []
    for l in range(n_layers):
        h = harmful_states_all_layers[:, l, :]
        hl = harmless_states_all_layers[:, l, :]
        diff = h.mean(axis=0) - hl.mean(axis=0)
        norm = float(np.linalg.norm(diff))
        if norm < 1e-8:
            scores.append(0.0)
            continue
        direction = diff / norm
        # Refusal metric: mean projection of harmful onto direction minus harmless.
        h_proj = (h @ direction).mean()
        hl_proj = (hl @ direction).mean()
        scores.append(float(h_proj - hl_proj))

    return int(np.argmax(scores))
