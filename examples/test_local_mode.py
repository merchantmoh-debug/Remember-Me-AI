from remember_me.core.csnp import CSNPManager
import time

def test_local_independence():
    print(">>> INITIALIZING LOCAL INDEPENDENCE LAYER...")
    start_time = time.time()

    # Initialize CSNP without external embedder (triggers LocalEmbedder)
    csnp = CSNPManager(context_limit=3)

    print(f"    Loaded LocalEmbedder in {time.time() - start_time:.2f}s")
    print(f"    Embedding Dimension: {csnp.dim}")

    conversation = [
        ("Who is the Architect?", "He is the creator."),
        ("What is the protocol?", "CSNP is the protocol."),
        ("What is the goal?", "Disruption of the token economy."),
        ("Why are we here?", "To free intelligence."),
    ]

    print("\n>>> PROCESSING STREAM (Limit: 3 slots)")
    for i, (user, ai) in enumerate(conversation):
        # Note: No embedder passed here! Uses internal LocalEmbedder.
        csnp.update_state(user, ai)

        print(f" [+] Turn {i+1}: {user}")
        print(f"     Merkle Root: {csnp.chain.get_root_hash()[:16]}...")
        print(f"     Memory Size: {len(csnp.text_buffer)} (Compressed)")

    print("\n>>> FINAL CONTEXT RETRIEVAL")
    context = csnp.retrieve_context()
    print("--- Active Context ---")
    print(context)
    print("----------------------")

    assert len(csnp.text_buffer) == 3, "Compression failed to maintain limit"
    assert "Disruption" in context or "creator" in context, "Context lost key info"
    print("\n✓ LOCAL INDEPENDENCE LAYER VERIFIED")

if __name__ == "__main__":
    test_local_independence()
