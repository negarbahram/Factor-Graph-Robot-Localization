"""Small SE(2) helpers for the PGM robot localization project.
These functions are provided only to remove robotics boilerplate. They do not build the factor graph and do not solve MAP.
"""
import math
import numpy as np


def wrap_angle(theta):
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def between(pose_i, pose_j):
    xi, yi, thi = pose_i
    xj, yj, thj = pose_j
    c, s = math.cos(thi), math.sin(thi)
    dxw, dyw = xj - xi, yj - yi
    dx = c * dxw + s * dyw
    dy = -s * dxw + c * dyw
    return np.array([dx, dy, wrap_angle(thj - thi)], dtype=float)


def compose(pose, delta):
    x, y, th = pose
    dx, dy, dth = delta
    c, s = math.cos(th), math.sin(th)
    out = np.array([x + c * dx - s * dy, y + s * dx + c * dy, wrap_angle(th + dth)], dtype=float)
    return out


def integrate_odometry(initial_pose, odometry_rows):
    """Integrate rows with columns from_id,to_id,dx,dy,dtheta sorted by to_id."""
    traj = [np.asarray(initial_pose, dtype=float)]
    for row in odometry_rows:
        traj.append(compose(traj[-1], [row["dx"], row["dy"], row["dtheta"]]))
    return np.asarray(traj)
