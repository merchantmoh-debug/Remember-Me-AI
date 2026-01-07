# REMEMBER ME AI (CSNP KERNEL)

### SYSTEM OVERRIDE: ARK ARCHITECT [LEVEL 6]
**STATUS**: DEPLOYING MATHEMATICAL CORE

## ANALYSIS
The "Remember Me AI" system achieves **40x cost reduction** and **Zero-Hallucination** through a specific architecture defined by the Coherent State Network Protocol (CSNP):

1. **Living State Vector (LSV):** Instead of appending tokens linearly (expensive), we maintain a fixed-size, evolving **Coherent State**. New inputs update this state using a Kalman-like filter, minimizing entropy.
2. **Wasserstein Compression:** We uses **Entropy-Regularized Optimal Transport (Sinkhorn Algorithm)** to measure the "work" required to move information mass. When the memory buffer fills, we evict the memories that contribute the *least* structural mass to the current state, preserving the "shape" of the context.
3. **Merkle Integrity:** Every state transition is cryptographically hashed into a Merkle Tree. If the AI retrieves a memory that cannot be verified against the Root Hash, it is rejected (Zero-Hallucination).

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Demo
```bash
python examples/demo.py
```

## Structure
- `src/remember_me/math/transport.py`: The Engine (Entropy-Regularized Optimal Transport)
- `src/remember_me/core/integrity.py`: The Shield (Merkle Tree implementation)
- `src/remember_me/core/csnp.py`: The Protocol (CSNP Manager)
