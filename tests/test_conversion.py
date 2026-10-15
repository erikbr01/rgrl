"""Unit tests for rgrl.conversion.plan_transforms — the source/target
shape_meta diff engine that drives rgrl.data.load_store's conversion."""

import numpy as np
import pytest

from rgrl.conversion import plan_transforms


def _meta(obs: dict, action_shape: list[int]) -> dict:
    return {"obs": obs, "action": {"shape": action_shape}}


def test_identity_diff_casts_to_float32_and_passes_through():
    meta = _meta({"pos": {"shape": [3], "type": "low_dim"}}, [7])
    transform, image_keys, source_key_of = plan_transforms(meta, meta, rename={})
    assert image_keys == []
    assert source_key_of == {"pos": "pos"}

    ep = {"pos": np.array([[1.0, 2.0, 3.0]], dtype=np.float64), "action": np.zeros((1, 7), dtype=np.float64)}
    out = transform(ep)
    assert out["pos"].dtype == np.float32
    assert out["action"].dtype == np.float32
    np.testing.assert_allclose(out["pos"], [[1.0, 2.0, 3.0]])


def test_rename_maps_source_key_to_target_key():
    source = _meta({"eef_pos": {"shape": [3], "type": "low_dim"}}, [7])
    target = _meta({"robot0_eef_pos": {"shape": [3], "type": "low_dim"}}, [7])
    transform, _image_keys, source_key_of = plan_transforms(
        source, target, rename={"eef_pos": "robot0_eef_pos"}
    )
    assert source_key_of == {"robot0_eef_pos": "eef_pos"}

    ep = {"eef_pos": np.array([[1.0, 2.0, 3.0]]), "action": np.zeros((1, 7))}
    out = transform(ep)
    assert "robot0_eef_pos" in out and "eef_pos" not in out


def test_channel_order_mismatch_reverses_last_axis():
    source = _meta({"cam": {"shape": [4, 4, 3], "type": "rgb", "channel_order": "bgr"}}, [7])
    target = _meta({"cam": {"shape": [4, 4, 3], "type": "rgb", "channel_order": "rgb"}}, [7])
    transform, image_keys, _ = plan_transforms(source, target, rename={})
    assert image_keys == ["cam"]

    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    ep = {"cam": img[None], "action": np.zeros((1, 7))}
    out = transform(ep)
    np.testing.assert_array_equal(out["cam"][0], img[..., ::-1])
    assert out["cam"].dtype == np.uint8  # images are never float32-cast


def test_quat_convention_mismatch_reorders_components():
    source = _meta({"q": {"shape": [4], "type": "low_dim", "quat_convention": "xyzw"}}, [7])
    target = _meta({"q": {"shape": [4], "type": "low_dim", "quat_convention": "wxyz"}}, [7])
    transform, _, _ = plan_transforms(source, target, rename={})

    ep = {"q": np.array([[1.0, 2.0, 3.0, 4.0]]), "action": np.zeros((1, 7))}  # x, y, z, w
    out = transform(ep)
    np.testing.assert_allclose(out["q"], [[4.0, 1.0, 2.0, 3.0]])  # w, x, y, z


def test_unresolvable_shape_mismatch_raises():
    source = _meta({"pos": {"shape": [3], "type": "low_dim"}}, [7])
    target = _meta({"pos": {"shape": [4], "type": "low_dim"}}, [7])
    with pytest.raises(ValueError, match="shape mismatch"):
        plan_transforms(source, target, rename={})


def test_image_downscale_is_planned():
    source = _meta({"cam": {"shape": [8, 8, 3], "type": "rgb"}}, [7])
    target = _meta({"cam": {"shape": [4, 4, 3], "type": "rgb"}}, [7])
    transform, image_keys, _ = plan_transforms(source, target, rename={})
    assert image_keys == ["cam"]
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[:, :4] = 255  # top half white, bottom half black
    out = transform({"cam": frames, "action": np.zeros((2, 7), np.float32)})
    assert out["cam"].shape == (2, 4, 4, 3)
    assert out["cam"].dtype == np.uint8
    # area-averaged downscale keeps the top/bottom split rather than smearing it
    assert out["cam"][0, :2].min() == 255 and out["cam"][0, 2:].max() == 0


def test_image_upscale_raises():
    source = _meta({"cam": {"shape": [4, 4, 3], "type": "rgb"}}, [7])
    target = _meta({"cam": {"shape": [8, 8, 3], "type": "rgb"}}, [7])
    with pytest.raises(ValueError, match="upscale"):
        plan_transforms(source, target, rename={})


def test_planned_rewrites_flags_only_byte_changes():
    from rgrl.conversion import planned_rewrites

    same = _meta({"cam": {"shape": [8, 8, 3], "type": "rgb"}}, [7])
    assert planned_rewrites(same, same, rename={}) == []
    smaller = _meta({"cam": {"shape": [4, 4, 3], "type": "rgb"}}, [7])
    assert planned_rewrites(same, smaller, rename={}) == ["cam"]


def test_action_shape_mismatch_raises():
    source = _meta({}, [7])
    target = _meta({}, [10])
    with pytest.raises(ValueError, match="rotation-representation"):
        plan_transforms(source, target, rename={})


def test_missing_target_key_raises():
    source = _meta({"pos": {"shape": [3], "type": "low_dim"}}, [7])
    target = _meta({"other": {"shape": [3], "type": "low_dim"}}, [7])
    with pytest.raises(ValueError, match="no matching source key"):
        plan_transforms(source, target, rename={})
