"""Forward-hook activation capture for a running policy.

`InferenceConfig.capture_layers` names dotted submodule paths (as in
`nn.Module.named_modules()`, e.g. "enc.encoders.agentview_image" or
"unet.mid_modules.1") whose output should be captured on every
predict_action() call — needed by the policy_graphs project to inspect a
policy's internal representations during rollout, not just its actions.

Wired in at two points:
  - rgrl/inference/policy_server.py: captures during /infer, returns them
    alongside the action (prefixed `act:<layer>` in the wire payload).
  - rgrl/inference/policy_client.py: LocalPolicy captures directly in-process;
    HttpPolicy unpacks the server's `act:<layer>` keys. Both expose the
    result as `.last_activations` after get() — None if no layers were
    requested.
"""

from __future__ import annotations

import numpy as np

ACTIVATION_KEY_PREFIX = "act:"


class ActivationCapture:
    """Registers a forward hook on each requested submodule; `pop()` returns
    everything captured since the last call, as numpy arrays."""

    def __init__(self, policy: nn.Module, layers: list[str]):
        self.layers = list(layers)
        self._activations: dict[str, "torch.Tensor"] = {}
        self._handles = []
        for name in self.layers:
            try:
                module = policy.get_submodule(name)
            except AttributeError as e:
                available = [n for n, _ in policy.named_modules() if n]
                raise ValueError(
                    f"capture_layers: {name!r} is not a submodule of this policy. "
                    f"Available submodules: {available}"
                ) from e
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._activations[name] = output.detach()

        return hook

    def pop(self) -> dict[str, np.ndarray]:
        """Return and clear the activations captured since the last pop()."""
        out = {k: v.float().cpu().numpy() for k, v in self._activations.items()}
        self._activations.clear()
        return out

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
