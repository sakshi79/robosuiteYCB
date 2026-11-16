"""
Checks for YCB objects in PickPlace (PickPlaceCan(object_type=...)).

Usage (from anywhere, in the env robosuite is installed in):
    MUJOCO_GL=egl python tests/ycb/test_pick_place_ycb.py
    MUJOCO_GL=egl pytest tests/ycb/test_pick_place_ycb.py
"""
import numpy as np
import robosuite as suite
from robosuite.models.objects import CanObject, CanVisualObject, YCBObject, YCBVisualObject

# small, tall, and wide objects: wide ones used to fail placement when all 4 objects had to fit in bin1
YCB_OBJECTS = ["011_banana", "003_cracker_box", "006_mustard_bottle", "025_mug", "035_power_drill", "029_plate"]
MARKERS = ["VisualMilk", "VisualBread", "VisualCereal", "VisualCan"]


def make_env(object_type, use_camera_obs=False, **kwargs):
    return suite.make(
        "PickPlaceCan", robots="Panda", object_type=object_type, has_renderer=False,
        has_offscreen_renderer=use_camera_obs, use_camera_obs=use_camera_obs,
        camera_names="agentview", camera_heights=64, camera_widths=64, **kwargs,
    )


def test_default_is_can():
    env = make_env(None)
    assert isinstance(env.objects[env.object_id], CanObject)
    assert env.object_model is None
    env.close()


def test_ycb_objects_reset_and_step():
    for obj in YCB_OBJECTS:
        env = make_env(obj)
        assert isinstance(env.objects[env.object_id], YCBObject), obj
        for _ in range(20):  # placement is random; repeat to catch intermittent RandomizationError
            env.reset()
            obs, r, done, info = env.step(np.zeros(env.action_dim))
            # active object keeps the "Can" name, so its observations and bin are the can's
            assert "Can_pos" in obs, obj
            # it starts in bin1, so the task isn't already solved
            assert not env._check_success(), obj
            # the other three are cleared out of the scene
            for name in ["Milk", "Bread", "Cereal"]:
                assert env.sim.data.body_xpos[env.obj_body_id[name]][0] > 5, (obj, name)
        env.close()


def test_success_in_can_bin():
    """Teleporting the YCB object into the can's bin (gripper far away) counts as success."""
    env = make_env("011_banana")
    env.reset()
    target = env.target_bin_placements[env.object_id].copy()
    target[2] += 0.05
    obj = env.objects[env.object_id]
    env.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([target, [1, 0, 0, 0]]))
    for _ in range(20):
        env.step(np.zeros(env.action_dim))
    assert env._check_success()
    env.close()


def test_object_model_requires_single_object_mode():
    try:
        suite.make("PickPlace", robots="Panda", object_model="011_banana", has_renderer=False,
                   has_offscreen_renderer=False, use_camera_obs=False)
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError for object_model without single_object_mode=2")


def test_ycb_ghost_marker():
    """A YCB object gets a see-through gray, joint- and collision-free ghost of itself as the can's target marker."""
    env = make_env("011_banana", use_camera_obs=True)
    env.reset()
    ghost = env.visual_objects[env.object_id]
    assert isinstance(ghost, YCBVisualObject) and ghost.ycb_id == "011_banana"
    assert ghost.joints == []
    m = env.sim.model
    bid = env.obj_body_id["VisualCan"]
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
    assert geoms, "ghost has no geoms"
    for g in geoms:
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0  # no collision
        np.testing.assert_allclose(m.geom_rgba[g], [0.8, 0.8, 0.8, 0.3], atol=1e-6)
    # the textured renderer must keep the ghost gray, not texture it
    tm = env._ycb_textured_renderer.model
    for g in geoms:
        np.testing.assert_allclose(tm.geom_rgba[g], [0.8, 0.8, 0.8, 0.3], atol=1e-6)
        assert tm.geom_matid[g] == -1
    # the real banana is still textured
    real_bid = env.obj_body_id["Can"]
    assert any(tm.geom_matid[g] >= 0 for g in range(tm.ngeom) if tm.geom_bodyid[g] == real_bid)
    # the other three ghosts are unchanged
    for i, name in enumerate(["Milk", "Bread", "Cereal"]):
        assert not isinstance(env.visual_objects[i], YCBVisualObject), name
    env.close()


def test_non_ycb_model_keeps_default_ghost():
    env = make_env("lemon")
    assert isinstance(env.visual_objects[env.object_id], CanVisualObject)
    env.close()


def test_textured_markers_match_sim():
    """The textured renderer must draw the 4 target markers where the sim has them, not at the origin."""
    for hard_reset in [True, False]:
        env = make_env("011_banana", use_camera_obs=True, hard_reset=hard_reset)
        for _ in range(2):
            env.reset()
            renderer = env._ycb_textured_renderer
            renderer.render(env, "agentview")
            for name in MARKERS:
                bid = env.obj_body_id[name]
                np.testing.assert_allclose(renderer.data.xpos[bid], env.sim.data.body_xpos[bid], atol=1e-9,
                                           err_msg=f"{name} hard_reset={hard_reset}")
        env.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__, flush=True)
    print(f"all {len(tests)} passed")
