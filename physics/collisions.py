from __future__ import annotations

import mujoco


def has_ball_surface_contact(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, str | None]:
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    for idx in range(data.ncon):
        contact = data.contact[idx]
        if ball_id not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == ball_id else contact.geom1
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other)
        if name in {"floor", "low_platform", "high_platform", "bridge_platform"}:
            return True, name
    return False, None
