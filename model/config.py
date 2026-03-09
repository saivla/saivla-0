"""
Sai0-VLA Model Configuration

Public configuration schema for the Sai0-VLA architecture.
Hyperparameter values shown here are illustrative defaults;
actual values used in our released checkpoints may differ.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class VLMConfig:
    """
    Vision-Language Model backbone configuration.

    Sai0-VLA supports pluggable VLM backbones. The backbone
    provides multi-scale hidden representations that are
    consumed by the action head via adaptive feature routing.
    """

    model_type: str = "eagle2_5_vl"
    """Backbone family: ``eagle2_5_vl`` | ``qwen3_vl`` | ``siglip_so400m``"""

    model_path: str = ""
    """HuggingFace model id or local path."""

    hidden_layers: List[int] = field(default_factory=lambda: [-3, -2, -1])
    """
    Layer indices for multi-scale feature extraction.

    We select representations from several transformer layers to capture
    both low-level spatial detail and high-level semantic information.
    Negative indices count from the last layer.
    """

    feature_projection: str = "adaptive_gate"
    """
    How multi-layer features are combined before entering the action head.
    Options: ``concat`` | ``adaptive_gate`` | ``learned_weighted_sum``
    """

    dtype: str = "bfloat16"
    image_size: int = 384
    """Native image resolution fed to the VLM (before internal tiling)."""

    rope_scaling: Optional[Dict[str, Any]] = None
    """Optional RoPE scaling config for long-context visual tokens."""


@dataclass
class ActionHeadConfig:
    """
    Action Head configuration.

    The action head transforms VLM hidden states + proprioceptive
    state into a chunk of future actions. Sai0-VLA's action head
    uses a variant of the Optimal Flow Transport (OFT) architecture
    with several custom modifications (see architecture.py).
    """

    head_type: str = "oft_v2"
    """Head variant: ``oft_v1`` | ``oft_v2`` | ``diffusion``"""

    hidden_dim: int = 1024
    """Internal hidden dimension of the action head transformer."""

    num_layers: int = 6
    """Number of transformer layers in the action head."""

    num_heads: int = 16
    """Number of attention heads per transformer layer."""

    action_dim: int = 7
    """Per-step action dimensionality (position + orientation + gripper)."""

    action_horizon: int = 16
    """Number of future action steps predicted per inference call."""

    proprio_dim: int = 8
    """Proprioceptive state dimensionality."""

    dropout: float = 0.05

    use_cross_attention: bool = True
    """
    If True, action queries attend to VLM features via cross-attention.
    If False, VLM features and action queries are concatenated.
    """

    use_spectral_norm: bool = True
    """Apply spectral normalization to linear layers for training stability."""

    flow_matching_steps: int = 0
    """
    Number of flow matching ODE steps at inference time (0 = direct regression).
    When > 0, the action head operates as a conditional flow matching model.
    """

    learned_action_queries: bool = True
    """Use learned positional queries for each action step in the horizon."""

    feature_conditioning: str = "film"
    """
    How proprioceptive state conditions the transformer.
    Options: ``concat`` | ``film`` | ``cross_attention`` | ``adaptive_norm``
    """


@dataclass
class Sai0VLAConfig:
    """
    Top-level Sai0-VLA configuration.

    Example::

        config = Sai0VLAConfig(
            vlm=VLMConfig(model_type="eagle2_5_vl"),
            action_head=ActionHeadConfig(head_type="oft_v2"),
        )
        model = Sai0VLAModel(config)
    """

    vlm: VLMConfig = field(default_factory=VLMConfig)
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)

    normalize_actions: bool = True
    """Whether to apply dataset-specific (de)normalization."""

    normalize_state: bool = True
    """Whether to normalize proprioceptive state before feeding to action head."""

    state_representation: str = "axis_angle"
    """
    Robot state orientation format.
    ``quaternion`` | ``axis_angle`` | ``rotation_6d``
    Conversion is handled automatically based on this setting.
    """

    temporal_ensemble: bool = False
    """
    Enable temporal action ensembling across overlapping chunks.
    Reduces jitter at the cost of slight delay.
    """

    ensemble_weights: str = "exponential_decay"
    """Weighting scheme for temporal ensembling: ``uniform`` | ``exponential_decay``"""
