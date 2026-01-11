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

        self.model_name = model_name
        self.model = None
        self._dim = None # Lazy load dimension

    @property
    def dim(self) -> int:
        """
        Returns the embedding dimension. Triggers model load if not known.
        """
        if self._dim is None:
             # If we are using the default model, we know it's 384.
             # But to be safe and support all models, we trigger the load.
             self._ensure_model_loaded()
        return self._dim

    @dim.setter
    def dim(self, value: int):
        self._dim = value

    def _ensure_model_loaded(self):
        if self.model is None:
            print(f"⚡ Bolt: Lazy loading embedding model: {self.model_name} on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self._dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Embeds text into a torch tensor [N, D].
        """
        self._ensure_model_loaded()

        if isinstance(text, str):
            text = [text]

        embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device)

        # Ensure we return [N, D]
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        return embeddings
