import sapien
import numpy as np
from scipy.spatial.transform import Rotation


def cam_and_lookat_pos_to_camera_R_and_T(cam_pos, look_at_point, yaw=0):
    cam_pos = np.array(cam_pos)
    look_at_point = np.array(look_at_point)
    cam_rel_pos = cam_pos - look_at_point
    forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    # Apply yaw to the camera
    quat = np.concatenate([[np.cos(yaw / 2.)], forward * np.sin(yaw / 2.)])
    rot = Rotation.from_quat(quat, scalar_first=True)
    left = rot.apply(left)

    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos

    scene = sapien.Scene()
    camera = scene.add_camera(
            name='camera',
            width=100,
            height=100,
            fovy=np.deg2rad(45),
            near=0.1,
            far=10,
    )
    camera.entity.set_pose(sapien.Pose(mat44))
    scene.update_render()
    mat44 = camera.get_model_matrix()

    # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    mat44[:3, 1:3] *= -1
    # get the world-to-camera transform and set R, T
    w2c = np.linalg.inv(mat44)
    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    return R, T

def r_theta_phi_and_lookat_pos_to_camera_R_and_T(r, theta, phi, look_at_point, yaw=0):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    cam_pos = np.array([x, y, z]) + np.array(look_at_point)

    return cam_and_lookat_pos_to_camera_R_and_T(cam_pos, look_at_point, yaw)

def compute_perpendicular_foot_direct(point1, vector1, point2, vector2):
    a = np.inner(vector1, vector2)
    b = np.inner(vector1, vector1)
    c = np.inner(vector2, vector2)
    d = np.inner((point2-point1), vector1)
    e = np.inner((point2-point1), vector2)
    if a == 0:
        t1 = d/b
        t2 = -e/c
        mod = 'perpendicular'
    elif ( a * a - b * c) == 0:
        t1 = 0
        t2 = -d/a
        mod = 'parallel'
    else:
        t1 = ( a * e - c * d) / ( a * a - b * c)
        t2 = b/a*t1-d/a
        mod = 'common'
    point1_tem = point1 + vector1 * t1
    point2_tem = point2 + vector2 * t2
    mid = (point1_tem + point2_tem) / 2.0
    return t1, t2, mid

def calculate_spherical_coord(lookat_pos, curr_pos):
    r = np.linalg.norm(curr_pos - lookat_pos)
    theta = np.arccos((curr_pos - lookat_pos)[2] / r)
    phi = np.arccos((curr_pos - lookat_pos)[0] / np.clip(r * np.sin(theta), 1e-4, 1e2))
    if np.linalg.norm(np.sin(phi) - (curr_pos - lookat_pos)[1] / np.clip(r * np.sin(theta), 1e-4, 1e2)) > 1e-3:
        phi = -phi
    return r, theta, phi

def camera_R_and_T_to_r_theta_phi_and_lookat_pos(w2c, w2c_1):
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    R_1 = w2c_1[:3, :3]
    t_1 = w2c_1[:3, 3]
    line_dir = R[:3, 2]
    line_dir_1 = R_1[:3, 2]
    _, _, lookat_pos = compute_perpendicular_foot_direct(t, line_dir, t_1, line_dir_1)
    r, theta, phi = calculate_spherical_coord(lookat_pos, t)
    r_1, theta_1, phi_1 = calculate_spherical_coord(lookat_pos, t_1)
    return r, theta, phi, r_1, theta_1, phi_1, lookat_pos
