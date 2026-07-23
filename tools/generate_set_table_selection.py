#!/usr/bin/env python3
"""Reproduce and validate the checked-in set-table episode manifests.

The source store must contain ``tasks.json`` mapping physical episode index to
task name. The original seed-42 selection sampled all task episodes, after
which demos known to be unsuccessful were replaced by successful demos from
the same task. Those explicit replacements live in each manifest's provenance,
so the complete result is reproducible and auditable in this repository.

By default this command checks that regeneration matches the committed files.
Pass ``--write`` to update their episode lists while preserving provenance.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RIGHT = "prepare plate - milk right"
MIDDLE = "prepare plate - milk middle"
LEFT = "prepare plate - milk left"


def load_task_episodes(store_path: Path) -> dict[str, list[int]]:
    tasks_path = store_path / "tasks.json"
    if not tasks_path.exists():
        raise FileNotFoundError(f"{tasks_path} does not exist")
    task_to_episodes: dict[str, list[int]] = defaultdict(list)
    for episode, task in json.loads(tasks_path.read_text()).items():
        task_to_episodes[str(task)].append(int(episode))
    return {task: sorted(episodes) for task, episodes in task_to_episodes.items()}


def base_select(episodes: list[int], count: int, seed: int) -> list[int]:
    values = np.asarray(episodes)
    if count >= len(values):
        return sorted(values.tolist())
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(values, size=count, replace=False).tolist())


def extra_select(
    episodes: list[int],
    already_used: set[int],
    count: int,
    seed: int,
) -> list[int]:
    remaining = np.asarray([episode for episode in episodes if episode not in already_used])
    if count >= len(remaining):
        return sorted(remaining.tolist())
    rng = np.random.default_rng(seed)
    return sorted(rng.permutation(remaining)[:count].tolist())


def apply_replacements(
    selected: list[int],
    replacements: dict[str, int],
    task: str,
    task_of_episode: dict[int, str],
) -> list[int]:
    result = set(selected)
    for old_text, new_value in replacements.items():
        old, new = int(old_text), int(new_value)
        if old not in result:
            raise ValueError(f"replacement source {old} is not in the generated {task} selection")
        if new in result:
            raise ValueError(f"replacement target {new} is already in the generated {task} selection")
        if task_of_episode.get(new) != task:
            raise ValueError(
                f"replacement {old}->{new} crosses tasks: expected {task!r}, "
                f"got {task_of_episode.get(new)!r}"
            )
        result.remove(old)
        result.add(new)
    return sorted(result)


def load_manifests(manifest_dir: Path) -> dict[str, dict]:
    names = (
        "set_table_baseline",
        "set_table_plusleft",
        "set_table_plusmiddle",
        "set_table_plusright",
    )
    return {
        name: json.loads((manifest_dir / f"{name}.json").read_text())
        for name in names
    }


def generate(
    task_to_episodes: dict[str, list[int]],
    manifests: dict[str, dict],
    seed: int,
) -> dict[str, list[int]]:
    task_of_episode = {
        episode: task
        for task, episodes in task_to_episodes.items()
        for episode in episodes
    }

    raw_right = base_select(task_to_episodes[RIGHT], 60, seed)
    raw_middle = base_select(task_to_episodes[MIDDLE], 60, seed)
    raw_left = base_select(task_to_episodes[LEFT], 60, seed)
    # The follow-up candidates were generated from the original base draw;
    # unsuccessful demos were curated afterward.
    raw_extra_right = extra_select(
        task_to_episodes[RIGHT], set(raw_right), 60, seed
    )
    raw_extra_middle = extra_select(
        task_to_episodes[MIDDLE], set(raw_middle), 60, seed
    )

    replacements = {
        name: manifest["provenance"].get("success_replacements", {})
        for name, manifest in manifests.items()
    }
    base_right = apply_replacements(
        raw_right,
        replacements["set_table_baseline"],
        RIGHT,
        task_of_episode,
    )
    # Baseline replacements only belong to the right task in the checked-in
    # selection. Keep this assertion explicit if future curation changes it.
    baseline_replacement_sources = {
        int(value) for value in replacements["set_table_baseline"]
    }
    if any(task_of_episode[value] != RIGHT for value in baseline_replacement_sources):
        raise ValueError("baseline success replacements currently assume right-task sources")

    baseline = sorted(base_right + raw_middle)
    left_additions = apply_replacements(
        raw_left,
        replacements["set_table_plusleft"],
        LEFT,
        task_of_episode,
    )
    middle_additions = apply_replacements(
        raw_extra_middle,
        replacements["set_table_plusmiddle"],
        MIDDLE,
        task_of_episode,
    )
    right_additions = apply_replacements(
        raw_extra_right,
        replacements["set_table_plusright"],
        RIGHT,
        task_of_episode,
    )
    return {
        "set_table_baseline": baseline,
        "set_table_plusleft": left_additions,
        "set_table_plusmiddle": middle_additions,
        "set_table_plusright": right_additions,
    }


def verify(
    generated: dict[str, list[int]],
    manifests: dict[str, dict],
    task_to_episodes: dict[str, list[int]],
) -> None:
    known = {episode for episodes in task_to_episodes.values() for episode in episodes}
    baseline = generated["set_table_baseline"]
    if len(baseline) != 120 or len(set(baseline)) != 120:
        raise ValueError("baseline must contain 120 unique episodes")

    additions = []
    for name, expected in generated.items():
        field = "episode_ids" if name == "set_table_baseline" else "add_episode_ids"
        actual = manifests[name][field]
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ValueError(
                f"{name} differs from regeneration; missing={missing}, extra={extra}"
            )
        if len(actual) != len(set(actual)):
            raise ValueError(f"{name} contains duplicate episode IDs")
        if unknown := sorted(set(actual) - known):
            raise ValueError(f"{name} contains unknown episode IDs: {unknown}")
        if name != "set_table_baseline":
            additions.append(set(actual))

    for index, left in enumerate(additions):
        for right in additions[index + 1 :]:
            if overlap := sorted(left & right):
                raise ValueError(f"follow-up additions overlap: {overlap}")


def write_manifests(
    generated: dict[str, list[int]],
    manifests: dict[str, dict],
    manifest_dir: Path,
) -> None:
    for name, values in generated.items():
        field = "episode_ids" if name == "set_table_baseline" else "add_episode_ids"
        manifests[name][field] = values
        path = manifest_dir / f"{name}.json"
        path.write_text(json.dumps(manifests[name], indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("configs/selections"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manifests = load_manifests(args.manifest_dir)
    task_to_episodes = load_task_episodes(args.store)
    generated = generate(task_to_episodes, manifests, args.seed)
    verify(generated, manifests, task_to_episodes)
    if args.write:
        write_manifests(generated, manifests, args.manifest_dir)
        print(f"updated four manifests in {args.manifest_dir}")
    else:
        print("all four set-table manifests match seed-42 regeneration")


if __name__ == "__main__":
    main()
