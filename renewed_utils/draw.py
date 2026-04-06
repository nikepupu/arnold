import numpy as np
from isaacsim.util.debug_draw import _debug_draw
from pxr import Gf

def draw_ee_target(position, orientation, axis_length=0.1, thickness=3.0):
    """
    Draw a coordinate frame at the target EE pose in the viewport.
    
    Args:
        position: array-like [x, y, z] in meters
        orientation: quaternion [w, x, y, z]  (Isaac Sim convention)
        axis_length: length of each axis line in meters
        thickness: line thickness in pixels
    """
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()

    w, x, y, z = orientation
    quat = Gf.Quatd(w, x, y, z)

    origin = list(position)

    axes = [
        (Gf.Vec3d(1, 0, 0), (1.0, 0.0, 0.0, 1.0)),  # X = red
        (Gf.Vec3d(0, 1, 0), (0.0, 1.0, 0.0, 1.0)),  # Y = green
        (Gf.Vec3d(0, 0, 1), (0.0, 0.0, 1.0, 1.0)),  # Z = blue
    ]

    starts, ends, colors, sizes = [], [], [], []
    for local_axis, color in axes:
        world_axis = quat.Transform(local_axis) * axis_length
        end = [origin[0] + world_axis[0], origin[1] + world_axis[1], origin[2] + world_axis[2]]
        starts.append(origin)
        ends.append(end)
        colors.append(color)
        sizes.append(thickness)

    draw.draw_lines(starts, ends, colors, sizes)
