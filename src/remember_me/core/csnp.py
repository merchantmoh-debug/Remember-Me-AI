import torch
import torch.nn.functional as F
import json
import time
from typing import List, Any, Dict, Optional
from ..math.transport import WassersteinMetric
from .integrity import IntegrityChain
from .embedder import LocalEmbedder

class CSNPManager:
    """
    Coherent State Network Protocol (CSNP) Manager.

    This class replaces the standard "Context Window".
    Instead of appending tokens (Linear Cost), it maintains a fixed-size
    buffer and an evolving "Identity State".

    When the buffer is full, it uses Wasserstein Optimization to identify
    the 'Mass' of information and evicts the lowest-mass vectors relative
    to the current narrative trajectory.
    """

    def __init__(self, embedding_dim: int = 384, context_limit: int = 50, embedder: Optional[Any] = None):
        """
        Args:
            embedding_dim: Dimension of the embedding vectors (default 384 for all-MiniLM-L6-v2).
            context_limit: Number of memory slots before compression triggers.
            embedder: Optional embedding model. If None, uses LocalEmbedder.
        """
        self.dim = embedding_dim
        self.context_limit = context_limit

        # Mathematical Engines
        self.metric = WassersteinMetric()
        self.chain = IntegrityChain()

        # Local Independence Layer
        if embedder is None:
            self.embedder = LocalEmbedder()
            self.dim = self.embedder.dim
        else:
            self.embedder = embedder

        # The "Living State Vector" (LSV)
        # Represents the aggregate direction of the session
        self.identity_state = torch.zeros(1, self.dim)

        # Memory Buffer (Compressed Context)
        self.memory_bank = torch.empty(0, self.dim)
        self.text_buffer: List[str] = []

    def update_state(self, user_input: str, ai_response: str, embedding_model: Optional[Any] = None):
        """
        CSNP Update Cycle:
        1. Integrity: Hash interaction into Merkle Tree.
        2. Embed: Vectorize the interaction.
        3. Evolve: Update Identity State (Kalman-like update).
        4. Compress: If full, evict lowest-mass memories via Wasserstein.
        """
        # 1. Integrity
        turn_text = f"USER:{user_input}|AI:{ai_response}"
        self.chain.add_entry(turn_text)

        # 2. Embed
        # Use internal embedder if none provided
        model = embedding_model if embedding_model else self.embedder

        with torch.no_grad():
            new_emb = model(turn_text)
            if new_emb.dim() == 1:
                new_emb = new_emb.unsqueeze(0) # Ensure [1, D]

        # 3. Evolve Identity State (Exponential Moving Average / Kalman approx)
        # This allows the "Self" to drift slowly with the conversation
        alpha = 0.1
        if self.identity_state.abs().sum() == 0:
             self.identity_state = new_emb.clone()
        else:
             self.identity_state = (1 - alpha) * self.identity_state + alpha * new_emb
        self.identity_state = F.normalize(self.identity_state, p=2, dim=1)

        # 4. Update Buffer
        self.memory_bank = torch.cat([self.memory_bank, new_emb], dim=0)
        self.text_buffer.append(turn_text)

        # 5. Compression via Optimal Transport
        if self.memory_bank.shape[0] > self.context_limit:
            self._compress()

    def _compress(self):
        """
        Reduces memory size while preserving maximum information mass
        relative to the Identity State.
        """
        # Calculate Wasserstein Mass contribution of all memories to the Identity State
        scores = self.metric.compute_transport_mass(self.identity_state, self.memory_bank)

        # Keep top K indices (Highest Mass / Most relevant to current state)
        _, keep_indices = torch.topk(scores, k=self.context_limit)
        keep_indices, _ = torch.sort(keep_indices) # Maintain chronological order

        # Prune buffers
        self.memory_bank = self.memory_bank[keep_indices]
        self.text_buffer = [self.text_buffer[i] for i in keep_indices.tolist()]

        # Rebuild Integrity Chain for the compressed state (Optional, creates Checkpoint)
        # In this impl, we keep the full Merkle history for verification,
        # even if the embedding is evicted.

    def retrieve_context(self) -> str:
        """
        Returns the current Coherent State (Context) for injection into the LLM.
        Verifies integrity before returning.
        """
        valid_texts = []
        for text in self.text_buffer:
            if self.chain.verify(text):
                valid_texts.append(text)
            else:
                print(f"⛔ HALLUCINATION DETECTED: {text[:15]}... rejected by Merkle Chain.")

        return "\n".join(valid_texts)

    def export_state(self) -> str:
        """
        Exports the CSNP state token.
        """
        state = {
            "merkle_root": self.chain.get_root_hash(),
            "memory_count": len(self.text_buffer),
            "identity_vector_norm": self.identity_state.norm().item(),
            "protocol": "CSNP/v1"
        }
        return json.dumps(state, indent=2)
