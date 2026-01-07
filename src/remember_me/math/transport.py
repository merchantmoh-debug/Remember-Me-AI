import torch
import torch.nn.functional as F

class WassersteinMetric:
    """
    Implements Entropy-Regularized Optimal Transport (Sinkhorn Algorithm).

    Unlike Cosine Similarity (which measures angles), Wasserstein measures the
    'Work' required to transform the Memory Distribution into the Query Distribution.

    This allows us to quantify the 'Information Mass' of a memory fragment.
    """

    def __init__(self, epsilon: float = 0.1, max_iter: int = 100, tol: float = 1e-6):
        self.epsilon = epsilon # Entropic regularization (smoothness)
        self.max_iter = max_iter
        self.tol = tol

    def compute_cost_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes squared Euclidean distance cost matrix.
        C_ij = ||x_i - y_j||^2
        """
        # x: [N, D], y: [M, D] -> Cost: [N, M]
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        cost = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(cost, min=0.0)

    def compute_transport_mass(self,
                             query_state: torch.Tensor,
                             memory_bank: torch.Tensor) -> torch.Tensor:
        """
        Computes the Optimal Transport Plan to determine how much 'mass'
        each memory contributes to the current query state.

        Args:
            query_state: [1, D] The current Coherent State vector.
            memory_bank: [N, D] The buffer of memory vectors.

        Returns:
            mass_scores: [N] Relevance score for each memory.
        """
        device = query_state.device
        N = memory_bank.size(0)
        M = query_state.size(0) # Usually 1

        if N == 0:
            return torch.tensor([])

        # Cost Matrix C [N, M]
        C = self.compute_cost_matrix(memory_bank, query_state)

        # Gibbs Kernel K = exp(-C / epsilon)
        K = torch.exp(-C / self.epsilon)

        # Sinkhorn Iterations (Fixed Point)
        # Initialize marginals (uniform assumption)
        u = torch.ones(N, 1, device=device) / N
        v = torch.ones(M, 1, device=device) / M

        for _ in range(self.max_iter):
            u_prev = u.clone()

            # v = 1 / (K^T @ u)
            # Note: We normalize by N/M to keep mass balanced
            v = 1.0 / (torch.mm(K.t(), u) + 1e-9)

            # u = 1 / (K @ v)
            u = 1.0 / (torch.mm(K, v) + 1e-9)

            if torch.max(torch.abs(u - u_prev)) < self.tol:
                break

        # Transport Plan Gamma = diag(u) @ K @ diag(v)
        # We just need the row sums (mass sent FROM each memory)
        # Gamma_i = u_i * (K @ v)_i
        # Since we want relevance, we look at the mass coupling
        Gamma = u * (K * v.t())

        # Sum mass transported FROM each memory index (row sum)
        mass_scores = Gamma.sum(dim=1)
        return mass_scores
