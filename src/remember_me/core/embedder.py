import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union

class LocalEmbedder:
    """
    Provides local, cost-free embeddings using HuggingFace's Sentence Transformers.
    Default model: 'all-MiniLM-L6-v2' (Small, fast, effective).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading local embedding model: {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Embeds text into a torch tensor [N, D].
        """
        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)

        # Ensure we return [N, D]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings
