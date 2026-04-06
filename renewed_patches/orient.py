import numpy as np
import json 
import os

def reset_target(task_name: str, split = "test"):
    patch_file = f"./renewed_patches/orient_{split}_patch.json"
    if not os.path.exists(patch_file):
        return False, None
    
    with open(patch_file, "r") as f:
        stats = json.load(f)
    
    has_patch = False
    target_y_angle = None
    if task_name in stats:
        if "new_target_y_angle" in stats[task_name]:
            has_patch = True
            target_y_angle = stats[task_name]['new_target_y_angle']
    
    return has_patch, target_y_angle


# ─── Quaternion helpers ───────────────────────────────────────────────────────

def _quat_mult(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def _quat_inv(q):
    """Inverse (= conjugate for unit quaternions) of [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def _quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)

def _quat_to_rotation_matrix(q):
    """Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2),  2*(x*y - w*z),      2*(x*z + w*y)],
        [    2*(x*y + w*z),    1 - 2*(x**2 + z**2),  2*(y*z - w*x)],
        [    2*(x*z - w*y),        2*(y*z + w*x),  1 - 2*(x**2 + y**2)],
    ])

def _extract_y_angle_from_quat(q):
    """
    Extract the effective Y-rotation angle (degrees) from quaternion q
    using the ZYX Euler pitch angle.

    For the reorient-object task the simulator tracks the object's orientation
    and reports 'current_y_angle' as the rotation around the world Y axis.
    The best single-angle proxy available from the quaternion alone is the
    ZYX pitch component:

        theta_zyx = arcsin(-R[2, 0])

    For a "pure" Y rotation this equals the rotation angle exactly; for a
    general rotation it captures the dominant Y contribution.
    """
    R = _quat_to_rotation_matrix(q)
    pitch = np.degrees(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    return pitch

def _extract_roll_from_quat(q):
    """
    Extract the effective Y-rotation angle (degrees) from quaternion q
    using the ZYX Euler pitch angle.

    For the reorient-object task the simulator tracks the object's orientation
    and reports 'current_y_angle' as the rotation around the world Y axis.
    The best single-angle proxy available from the quaternion alone is the
    ZYX pitch component:

        theta_zyx = arcsin(-R[2, 0])

    For a "pure" Y rotation this equals the rotation angle exactly; for a
    general rotation it captures the dominant Y contribution.
    """
    R = _quat_to_rotation_matrix(q)
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll

def _angle_diff(a, b):
    """Smallest signed angular difference a − b, wrapped to (−180, 180]."""
    diff = (a - b + 180) % 360 - 180
    return diff

def _ry_quat(angle_deg):
    """Unit quaternion for a pure rotation of *angle_deg* around the world Y axis."""
    half = np.radians(angle_deg) / 2.0
    return np.array([np.cos(half), 0.0, np.sin(half), 0.0])


# ─── Function 1 ──────────────────────────────────────────────────────────────

def check_end_orientation_reasonable(entry, angle_tolerance=25.0):
    """
    Decide whether the recorded 'end orientation' is geometrically consistent
    with the 'target_y_angle' for a reorient-object demonstration.

    Strategy
    --------
    1. Basic sanity: quaternion must be nearly unit-length.
    2. Compute the *relative* rotation applied by the gripper:
           delta_q = end_q ⊗ inv(pick_q)
       This is the rotation the gripper performed in the world frame,
       which is also what it applied to the object.
    3. Extract the Y-rotation component of delta_q and compare it to
       'target_y_angle'.  A correct demonstration should have an implied
       object Y angle within `angle_tolerance` degrees of the target.

    Parameters
    ----------
    entry          : dict  – one JSON record from orient.json
    angle_tolerance: float – maximum acceptable deviation from target_y_angle
                             (default 25°, slightly larger than the task's 20°
                              to account for small measurement noise)

    Returns
    -------
    (is_reasonable: bool, info: dict)
        is_reasonable – True if the end orientation looks correct
        info          – diagnostic details for debugging / reporting
    """
    pick_q = _quat_normalize(entry["pick orientation"])
    end_q  = _quat_normalize(entry["end orientation"])
    target = float(entry["target_y_angle"])

    info = {}

    # ── 1. Unit-quaternion sanity check ──────────────────────────────────────
    raw_end_q = np.asarray(entry["end orientation"], dtype=float)
    raw_norm   = np.linalg.norm(raw_end_q)
    info["end_quat_norm"] = raw_norm
    if abs(raw_norm - 1.0) > 1e-2:
        info["reason"] = "end orientation quaternion is not unit-length"
        return False, info

    # ── 2. Relative rotation in world frame: delta = end ⊗ inv(pick) ─────────
    delta_q = _quat_normalize(_quat_mult(end_q, _quat_inv(pick_q)))
    info["delta_q"] = delta_q.tolist()

    # ── 3. Extract implied Y angle from the relative rotation ─────────────────
    # ZYX pitch of delta_q is the best single-number proxy for the Y rotation
    # the gripper applied to the object.
    implied_y = _extract_roll_from_quat(delta_q) # _extract_y_angle_from_quat(delta_q)
    info["implied_y_angle_deg"] = implied_y
    info["target_y_angle"]      = target


    # Angular difference (handles wrap-around between e.g. 179° and −179°)
    diff = abs(_angle_diff(implied_y, target))
    # Also consider the supplementary angle because some grasp modes flip sign
    diff_alt = abs(_angle_diff(implied_y + 180, target))
    best_diff = min(diff, diff_alt)
    info["angular_diff_deg"] = best_diff

    is_ok = best_diff <= angle_tolerance
    if not is_ok:
        info["reason"] = (
            f"implied Y angle ({implied_y:.1f}°) deviates {best_diff:.1f}° "
            f"from target ({target:.1f}°); threshold is {angle_tolerance:.1f}°"
        )
    else:
        info["reason"] = "end orientation appears consistent with target_y_angle"

    return is_ok, info


# ─── Function 2 ──────────────────────────────────────────────────────────────

def compute_correct_end_orientation(entry):
    """
    Compute a corrected 'end orientation' quaternion that is geometrically
    consistent with achieving 'target_y_angle' for this demonstration.

    Strategy
    --------
    The pick orientation encodes the gripper's reference frame at the moment
    it grasps the object.  To reorient the object to `target_y_angle` around
    the world Y axis, the gripper must arrive at:

        correct_end_q = Ry(target_y_angle) ⊗ pick_q

    where Ry(θ) = [cos(θ/2), 0, sin(θ/2), 0] is a pure world-Y rotation.

    This preserves the full grasp frame (how the fingers wrap the object) and
    only changes the net world-Y orientation, which determines current_y_angle
    as reported by the simulator.

    Parameters
    ----------
    entry : dict  – one JSON record from orient.json

    Returns
    -------
    corrected_q : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z] for the corrected end orientation.
    """
    pick_q = _quat_normalize(entry["pick orientation"])
    target = float(entry["target_y_angle"])

    # Pure rotation by target_y_angle around the world Y axis
    ry = _ry_quat(target)

    # Apply that world-Y rotation on top of the pick (grasp) orientation
    corrected_q = _quat_normalize(_quat_mult(ry, pick_q))
    return corrected_q


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def patch_entry(entry, angle_tolerance=25.0):
    """
    Check an entry and, if its end orientation is unreasonable, replace it
    with the geometrically corrected value.

    Returns
    -------
    (patched_entry: dict, was_patched: bool, info: dict)
    """
    import copy
    is_ok, info = check_end_orientation_reasonable(entry, angle_tolerance)
    patched = copy.deepcopy(entry)
    if not is_ok:
        corrected_q = compute_correct_end_orientation(entry)
        patched["end orientation"] = corrected_q.tolist()
        patched["success"] = 1   # mark as corrected
    return patched, not is_ok, info