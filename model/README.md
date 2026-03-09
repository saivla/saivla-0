# Sai0-VLA Model Architecture

## Overview

Sai0-VLA is a Vision-Language-Action (VLA) model for robotic manipulation.
It takes multi-view RGB images, a natural language instruction, and
proprioceptive state as input, and predicts a chunk of future robot actions.

```
Images + Instruction ──► VLM Backbone ──► Multi-Scale Features
                                              │
                                              ▼
                                    Adaptive Feature Routing
                                              │
                    Proprio ──► Encoder ──►    │
                                              ▼
                                      OFT Action Head
                                              │
                                              ▼
                                     Action Chunk (H × D)
```

## Key Components

### 1. VLM Backbone

We support multiple Vision-Language Model backbones:
- **Eagle 2.5 VL** (NVIDIA GR00T N1.5)
- **Qwen3-VL** (2B / 4B / 8B)

The backbone extracts **multi-scale features** from several internal
transformer layers. Lower layers capture fine-grained spatial detail
(object edges, grasp points), while higher layers encode task-level
semantics (object identity, spatial relations).

### 2. Adaptive Feature Routing

Rather than simple concatenation, we use a **learned gating mechanism**
to dynamically weight multi-layer features based on the current
observation context. This allows the model to emphasize spatial features
for precision grasping and semantic features for object selection.

### 3. OFT Action Head

Our action head is based on the **Optimal Flow Transport** (OFT) framework,
implemented as a transformer with the following features:

- **Learned action queries**: Positional queries for each timestep in the
  action horizon, refined via self-attention.
- **Cross-attention** to VLM features: Action queries attend to the fused
  visual-language representation.
- **FiLM conditioning**: Proprioceptive state modulates internal
  representations via Feature-wise Linear Modulation.
- **Dual inference mode**: Supports both direct L1 regression (fast) and
  conditional flow matching (higher quality, iterative).

### 4. Action Chunking

The model predicts **H = 16 future actions** in a single forward pass,
reducing VLM inference frequency by 16×. During deployment, all predicted
actions are executed before re-observing.

## Files

| File | Description |
|------|-------------|
| `config.py` | Configuration dataclasses |
| `architecture.py` | Architecture definition and module interfaces |

## Usage

The model architecture is provided for reference. For inference,
use our hosted API via the `sai0-vla-client` package:

```bash
pip install sai0-vla-client

# Run LIBERO evaluation
sai0-eval \
    --server https://api.sai0.ai \
    --api-key sk-xxx \
    --task-suite libero_spatial
```

See the [client README](../README.md) for full documentation.

## Citation

If you use Sai0-VLA in your research, please cite:

```bibtex
@article{sai0vla2026,
    title={Sai0-VLA: Vision-Language-Action Models with Optimal Flow Transport},
    author={Sai0 Team},
    year={2026},
}
```
