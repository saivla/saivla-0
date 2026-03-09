"""
Sai0-VLA Remote Inference Client

Lightweight wrapper that calls the cloud /v1/act endpoint for action predictions.
No model code included -- only depends on requests + numpy + Pillow.

Example::

    from sai0_vla_client import Sai0VLAClient  # after pip install
    # or: from client import Sai0VLAClient     # local dev
    client = Sai0VLAClient("https://api.sai0.ai", api_key="sk-xxx")
    actions = client.act(images=[img1, img2], state=state, instruction="pick up the mug")
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class Sai0VLAClient:
    """Client for communicating with the Sai0-VLA inference server."""

    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def act(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        state: Union[np.ndarray, List[float]],
        instruction: str,
        task_suite: Optional[str] = None,
    ) -> np.ndarray:
        """
        Send observations and get action predictions.

        Args:
            images: List of RGB images (np.ndarray uint8 HWC or PIL.Image).
            state: Robot state vector (raw values; server handles preprocessing).
            instruction: Task instruction text.
            task_suite: LIBERO task suite name (e.g. ``libero_spatial``).
                If None, the server uses its default engine.

        Returns:
            numpy array of shape (chunk_size, action_dim).
        """
        encoded = [self._encode_image(img) for img in images]
        state_list = state.tolist() if isinstance(state, np.ndarray) else list(state)

        payload = {
            "images": encoded,
            "state": state_list,
            "instruction": instruction,
            "image_format": "base64",
        }
        if task_suite is not None:
            payload["task_suite"] = task_suite

        resp = self._post("/v1/act", payload)
        return np.array(resp["actions"], dtype=np.float32)

    def health(self) -> Dict[str, Any]:
        """Health check."""
        return self._get("/health")

    def version(self) -> Dict[str, Any]:
        """Get server version info."""
        return self._get("/version")

    def metrics(self) -> Dict[str, Any]:
        """Get /v1 aggregated metrics."""
        return self._get("/v1/metrics")

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _encode_image(self, img: Union[np.ndarray, Image.Image]) -> str:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json_body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.server_url}{path}"
        resp = self._session.post(url, json=json_body, timeout=self.timeout)
        if resp.status_code == 429:
            raise RuntimeError(f"Rate limit exceeded: {resp.json().get('detail', '')}")
        if resp.status_code == 401:
            raise PermissionError(f"Authentication failed: {resp.json().get('detail', '')}")
        resp.raise_for_status()
        return resp.json()
