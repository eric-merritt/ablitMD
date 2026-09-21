"""SOM-based multi-directional ablation (SOM-MD).

Trains a Self-Organizing Map on harmful-prompt hidden states at a given layer,
then computes k unit directions from the k best neurons toward the harmless
centroid. These directions capture the refusal manifold structure that a single
difference-in-means direction misses.

Reference: "Multi-Directional Ablation" (SOM-MD) arXiv paper.
"""

import numpy as np


def hex_grid_neighbors(rows: int, cols: int) -> list[list[int]]:
    """Precompute the 6-neighborhood of every neuron on a hexagonal lattice.

    The lattice is laid out on a (rows, cols) rectangular array where even rows
    are shifted right by half a cell — the standard "odd-r" offset coordinate
    system for hex grids. Each neuron gets up to 6 neighbors; toroidal wrap in
    both axes so edge neurons still have full neighborhoods.

    Returns:
        List of length rows*cols, entry i is the list of neighbor indices for
        neuron i (row-major order).
    """
    # Axial offsets for an odd-r offset hex grid (even rows unshifted).
    # For even row r: neighbors are (r-1,c-1),(r-1,c),(r,c-1),(r,c+1),(r+1,c-1),(r+1,c)
    # For odd  row r: neighbors are (r-1,c),(r-1,c+1),(r,c-1),(r,c+1),(r+1,c),(r+1,c+1)
    def _even_offsets():
        return [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]

    def _odd_offsets():
        return [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

    neighbors: list[list[int]] = []
    for r in range(rows):
        offsets = _even_offsets() if r % 2 == 0 else _odd_offsets()
        for c in range(cols):
            nbrs = []
            for dr, dc in offsets:
                nr = (r + dr) % rows
                nc = (c + dc) % cols
                nbrs.append(nr * cols + nc)
            neighbors.append(nbrs)
    return neighbors


def train_som(
    data: np.ndarray,
    grid_shape: tuple[int, int] = (4, 4),
    n_iter: int = 1000,
    lr_init: float = 0.5,
    sigma_init: float = 2.0,
    seed: int = 42,
    hexagonal: bool = True,
) -> np.ndarray:
    """Train a Kohonen SOM on data using the standard sequential algorithm.

    Args:
        data: (n_samples, dim) harmful-prompt hidden states at layer l*.
        grid_shape: (rows, cols) for the SOM lattice.
        n_iter: number of training iterations.
        lr_init: initial learning rate (decays linearly to 0).
        sigma_init: initial neighbourhood radius (in grid units), decays to 0.
        seed: RNG seed for weight initialization.
        hexagonal: use a hexagonal lattice (6 neighbors) instead of the old
            rectangular toroidal grid (4 neighbors). Default True.

    Returns:
        weights: (rows*cols, dim) SOM weight matrix.
    """
    n_rows, n_cols = grid_shape
    n_neurons = n_rows * n_cols
    rng = np.random.default_rng(seed)

    # Precompute the neighborhood structure once — this is what makes hex fast:
    # each iteration only touches the BMU's neighbors instead of scanning all neurons.
    if hexagonal:
        neighbor_lists = hex_grid_neighbors(n_rows, n_cols)
    else:
        # Rectangular toroidal 4-neighborhood (up/down/left/right).
        neighbor_lists = []
        for i in range(n_rows):
            for j in range(n_cols):
                idx = i * n_cols + j
                neighbor_lists.append([
                    ((i - 1) % n_rows) * n_cols + j,
                    ((i + 1) % n_rows) * n_cols + j,
                    i * n_cols + ((j - 1) % n_cols),
                    i * n_cols + ((j + 1) % n_cols),
                ])

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

        # Update BMU + its neighbors (Gaussian kernel over lattice distance).
        # Include the BMU itself (distance 0) so it actually moves toward the sample.
        update_targets = [bmu] + neighbor_lists[bmu]
        for nbr in update_targets:
            lattice_dist = 0 if nbr == bmu else 1
            h = np.exp(-(lattice_dist ** 2) / (2.0 * max(sigma * sigma, 1e-8)))
            weights[nbr] += alpha * h * (sample - weights[nbr])

    return weights.astype(np.float32)


def compute_som_directions(
    harmful_states: np.ndarray,
    harmless_centroid: np.ndarray,
    grid_shape: tuple[int, int] = (4, 4),
    k: int | None = None,
    seed: int = 42,
    hexagonal: bool = True,
) -> list[np.ndarray]:
    """Train a SOM on harmful states and return unit directions from neurons toward
    the harmless centroid.

    Args:
        harmful_states: (n_harmful, dim) hidden states at layer l*.
        harmless_centroid: (dim,) mean of harmless-prompt hidden states at l*.
        grid_shape: SOM lattice dimensions.
        k: if given, return only the top-k neurons by BMU activation count
            (most representative of the harmful data). If None, return a
            direction for every neuron — the full candidate pool for BO search.
        seed: RNG seed for SOM initialization.
        hexagonal: use a hexagonal lattice (default True).

    Returns:
        List of unit vectors (dim,), each pointing from a SOM neuron toward
        the harmless centroid. Length is min(k, n_neurons) if k given, else n_neurons.
    """
    weights = train_som(
        harmful_states, grid_shape=grid_shape, seed=seed, hexagonal=hexagonal
    )
    n_neurons = len(weights)

    # Activation: how many harmful samples have this neuron as BMU.
    dists = np.linalg.norm(harmful_states[:, None, :] - weights[None, :, :], axis=2)
    bmu_counts = np.bincount(np.argmin(dists, axis=1), minlength=n_neurons)

    if k is not None:
        # Old behavior: top-k neurons by activation count.
        order = np.argsort(bmu_counts)[::-1][:k]
    else:
        # Full candidate pool: all neurons, ordered by activation so the most
        # representative ones come first (useful for logging / debugging).
        order = np.argsort(bmu_counts)[::-1]

    directions = []
    for neuron_idx in order:
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
