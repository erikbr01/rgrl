"""Print the ``expected_sha256`` audit lock for a data config's selection.

You cannot hand-write ``expected_sha256`` when authoring a config -- computing it
requires resolving the ``(task, seed, count)`` recipe against the source store's
``tasks.json``. So the flow is create-then-pin:

  1. Write the data config's ``selection:`` block with no ``expected_sha256``
     (it is optional; the selection resolves fine without it, just unlocked).
  2. Run this to resolve it and print the hash, then paste the printed
     ``expected_sha256: "..."`` line into the config's ``selection:`` block.

Pinning is optional but recommended: it turns a later silent change of the
resolved set (a store's ``curation.json`` edited, a numpy RNG version bump, an
edited seed/count) into a hard error instead of a quietly different dataset.

    uv run python -m rgrl.pin_selection data=set_table_baseline
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from rgrl.data import _episode_ids_sha256, _resolve_selection


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    data = OmegaConf.to_container(cfg.data, resolve=True)
    name = str(data.get("name", "selection"))
    selection = data.get("selection")
    if selection is None:
        print(f"data config {name!r} has no `selection:` block -- nothing to pin")
        return
    store_path = data.get("store_path")
    # Resolve ignoring any existing (possibly stale) lock, so this always reports
    # the hash the recipe currently produces.
    recipe = {k: v for k, v in selection.items() if k != "expected_sha256"}
    ids = _resolve_selection(recipe, Path(store_path) if store_path else None, name=name)
    sha = _episode_ids_sha256(ids)
    current = selection.get("expected_sha256")
    status = (
        "matches current lock" if current == sha
        else f"DIFFERS from current lock {current!r}" if current is not None
        else "no lock set yet"
    )
    print(f"{name}: resolved {len(ids)} episodes ({status})")
    print("paste this into the config's selection block:")
    print(f'  expected_sha256: "{sha}"')


if __name__ == "__main__":
    main()
