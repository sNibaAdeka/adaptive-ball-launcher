"""MuJoCo scene builder. Browser rendering uses the exact same arena dimensions."""
from __future__ import annotations

from xml.sax.saxutils import escape


def make_mjcf(ball: dict, environment: dict, physics: dict, *, mass: float | None = None, drag: float | None = None) -> str:
    width, length = environment["arena_width"], environment["arena_length"]
    platform_xml = "\n".join(
        f'<geom name="{escape(item["name"])}" type="box" pos="{item["position"][0]} {item["position"][1]} {item["position"][2]}" size="{item["size"][0]} {item["size"][1]} {item["size"][2]}" material="platform" friction="{0.9 if item["surface"] == "high_friction" else 0.25 if item["surface"] == "low_friction" else 0.52} 0.01 0.001"/> '
        for item in environment.get("platforms", [])
    )
    ball_mass, ball_drag = mass or ball["mass"], drag or ball["drag_coefficient"]
    # Density is only informative because the physical mass is explicit. Ball's collision radius is exact.
    return f'''<mujoco model="adaptive_ball_launcher">
  <compiler angle="degree" coordinate="local"/>
  <option timestep="{physics["timestep"]}" gravity="0 0 -{physics["gravity"]}" integrator="RK4"/>
  <default><geom condim="3" friction="{ball["surface_friction"]} 0.01 0.001" solref="0.006 1"/></default>
  <asset><material name="floor" rgba="0.11 0.14 0.18 1"/><material name="platform" rgba="0.22 0.35 0.45 1"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="{width/2} {length/2} 0.1" material="floor" friction="{ball["surface_friction"]} 0.01 0.001"/>
    {platform_xml}
    <body name="ball" pos="0 0 0"><freejoint name="ball_free"/>
      <geom name="ball" type="sphere" size="{ball["radius"]}" mass="{ball_mass}" rgba="1 0.78 0.24 1" friction="{ball["surface_friction"]} 0.01 0.001"/>
    </body>
  </worldbody>
</mujoco>'''
