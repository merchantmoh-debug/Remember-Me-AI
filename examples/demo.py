import torch
import torch.nn as nn
from remember_me.core.csnp import CSNPManager

# Mock Embedder (Deterministically maps text length to vectors for testing)
class MockEmbedder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, text):
        seed = len(text)
        torch.manual_seed(seed)
        return torch.randn(1, self.dim)

def main():
    print(">>> INITIALIZING CSNP KERNEL...")

    # 1. Initialize Embedder First (Dependency Injection)
    embedder = MockEmbedder(768)

    # 2. Initialize Manager with the embedder
    # We set context_limit=3 to force Wasserstein compression immediately
    csnp = CSNPManager(embedding_dim=768, context_limit=3, embedder=embedder)

    conversation = [
        ("Hello", "Hi there!"),
        ("My name is Mohamad", "Nice to meet you."),
        ("I like coding", "Coding is great."),
        ("What is the matrix?", "It is a system."),
        ("I also like math", "Math is the language of the universe."),
        ("What was my name?", "Your name is Mohamad.")
    ]

    print(f"\n>>> PROCESSING STREAM (Limit: 3 slots)")
    for i, (user, ai) in enumerate(conversation):
        csnp.update_state(user, ai)

        print(f" [+] Turn {i+1}: {user}")
        print(f"     Merkle Root: {csnp.chain.get_root_hash()[:16]}...")
        print(f"     Memory Size: {len(csnp.text_buffer)} (Compressed)")

    print("\n>>> FINAL STATE RETRIEVAL (Wasserstein Optimized)")
    context = csnp.retrieve_context()
    print("--- Active Context ---")
    print(context)
    print("----------------------")

    print("\n>>> SYSTEM STATUS")
    print(csnp.export_state())

if __name__ == "__main__":
    main()
