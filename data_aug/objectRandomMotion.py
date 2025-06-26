import os
import copy
import click
from mplib import Pose
import numpy as np
import torch
import h5py
from PIL import Image
from scipy.spatial.transform import Rotation

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_aug.util.util import save_rgb_images_to_video
from data_aug.util.robot_util import RobotUtil, INIT_QPOS
from data_aug.util.gaussian_util import GaussianModel, GaussianRenderer, Camera, get_robot_gaussian_at_qpos, transform_gaussian, get_T_for_rotating_around_an_axis
from data_aug.util.camera_util import cam_and_lookat_pos_to_camera_R_and_T
from data_aug.util.data_record_util import DataRecorder
from data_aug.util.augment_util import *


# @click.command()
# @click.option('--ref_demo_path', required=True, type=str)
# @click.option('--xy_step_str', required=True, type=str)
# @click.option('--augment_lighting', required=True, type=bool)
# @click.option('--augment_appearance', required=True, type=bool)
# @click.option('--augment_camera_pose', required=True, type=bool)
# @click.option('--output_path', required=True, type=str)
# @click.option('--image_size', required=True, type=int)
# @click.option('--save', required=True, type=bool)
# @click.option('--save_video', required=True, type=bool)

image_size = 256
save = True
save_video = True
ref_demo_path = 'data/source_demo/real_000000.h5'
xy_step_str = '[10, 10]'
augment_lighting = True
augment_appearance = False
augment_camera_pose = False
output_path = 'data/generated_demo/pick_68'

def pose_to_transformation_matrix(pose):
    # 构造变换矩阵
    transformation_matrix = np.eye(4)  # 初始化为 4x4 单位矩阵

    # 设置平移部分
    transformation_matrix[0:3, 3] = [pose[0], pose[1], pose[2]]

    # 四元数转旋转矩阵部分
    # 假设四元数是 w, x, y, z 的形式
    q_w = pose[3]
    q_x = pose[4]
    q_y = pose[5]
    q_z = pose[6]

    # 根据四元数构造旋转矩阵
    rotation_matrix = np.array([
        [1 - 2*q_y**2 - 2*q_z**2, 2*q_x*q_y - 2*q_z*q_w, 2*q_x*q_z + 2*q_y*q_w],
        [2*q_x*q_y + 2*q_z*q_w, 1 - 2*q_x**2 - 2*q_z**2, 2*q_y*q_z - 2*q_x*q_w],
        [2*q_x*q_z - 2*q_y*q_w, 2*q_y*q_z + 2*q_x*q_w, 1 - 2*q_x**2 - 2*q_y**2]
    ])

    # 将旋转矩阵放到变换矩阵的左上角
    transformation_matrix[0:3, 0:3] = rotation_matrix

    return transformation_matrix

Tz = np.array([[0,-1,0,0],
                [1, 0,0,0],
                [0, 0,1,0],
                [0, 0,0,1]], dtype=float)

Tx = np.array([[1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]], dtype=float)
Ty = np.array([[0, 0, 1, 0],
                [0, 1, 0, 0],
                [-1, 0, 0, 0],
                [0, 0, 0, 1]], dtype=float)

# 绕x轴旋转180度的变换矩阵
def rotation_x(angel):
    theta = np.radians(angel)  # 转换为弧度
    rotation_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return rotation_x
# 绕y轴旋转180度的变换矩阵
def rotation_y(angel):
    theta = np.radians(angel)  # 转换为弧度
    rotation_y = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    return rotation_y

# 绕z轴旋转180度的变换矩阵
def rotation_z(angel):
    theta = np.radians(angel)  # 转换为弧度
    rotation_z = np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return rotation_z

angleList = [rotation_x(90), rotation_x(180), rotation_x(270),
             rotation_y(90), rotation_y(180), rotation_y(270),
             rotation_z(90), rotation_z(180), rotation_z(270),]

def main(ref_demo_path, xy_step_str, augment_lighting, augment_appearance, augment_camera_pose, output_path, image_size, save, save_video):
    data_recorder = DataRecorder(save=save, data_path=output_path, image_shape=[image_size, image_size], save_video=save_video)

    np.random.seed(42)

    ref_demo_h5 = h5py.File(ref_demo_path, 'r')
    action_in_demo = ref_demo_h5['action'][()]
    qpos_in_demo = ref_demo_h5['obs']['qpos'][()]
    ee_pose_in_demo = ref_demo_h5['obs']['ee_pose'][()]
    origin_episode_length = action_in_demo.shape[0]
    key_frame_indices = []
    for i in range(1, origin_episode_length):
        if action_in_demo[i - 1][-1] != action_in_demo[i][-1]:
            key_frame_indices.append(i)
    key_frame_indices.append(origin_episode_length - 1)
    num_grasp_step = 3  # The number of steps required to close the gripper. It is usually 3 to 6.
    grasp_width = qpos_in_demo[:, -1].min()

    robot_util = RobotUtil()

    if not augment_camera_pose:
        camera_0, renderer_0, camera_1, renderer_1 = get_camera_and_renderer()

    object_gaussian_origin = GaussianModel(sh_degree=3)
    object_gaussian_origin.load_ply('data/gaussian/object/carrot.ply')
    # mango_gaussian = GaussianModel(sh_degree=3)
    # mango_gaussian.load_ply('/home/admin123/ssd/Projects/TRELLIS/output/mango/sample.ply')

    if not augment_appearance:
        table_gaussian = GaussianModel(sh_degree=3)
        table_gaussian.load_ply('data/gaussian/table.ply')
        # table_gaussian.load_ply('/home/admin123/ssd/Projects/GaussianEditor/dataset/bicycle/point_cloud/iteration_30000/point_cloud.ply')

    # augment object pose (align with pose randomization in env)
    x_range = np.linspace(-0.06, 0.24, eval(xy_step_str)[0])
    y_range = np.linspace(-0.22, 0.18, eval(xy_step_str)[1])
    z_range = [0.0]
    x, y, z = np.meshgrid(x_range, y_range, z_range)
    all_displacement = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    num_demo = all_displacement.shape[0]
    num_demo = 9
    # INIT_QPOS[] = INIT_QPOS[0]+0.2
    for demo_idx in range(num_demo):
        data_recorder.clean()

        displacement_pos = all_displacement[demo_idx]
        displacement_rot = np.random.uniform(-np.pi, np.pi)

        # generate robot behavior in new configuration
        ee_pose_list, qpos_list, action_list = [], [], []

        key_pose_0 = np.eye(4)
        key_pose_0[:3, 3] = ee_pose_in_demo[key_frame_indices[0]][:3, 3] + displacement_pos
        rot_euler = (Rotation.from_euler('Z', displacement_rot) * Rotation.from_matrix(ee_pose_in_demo[key_frame_indices[0]][:3, :3])).as_euler('XYZ')
        rot_euler[2] = normalize_and_adjust_angle(rot_euler[2])
        key_pose_0[:3, :3] = Rotation.from_euler('XYZ', rot_euler).as_matrix()

        key_pose_0_quat = Rotation.from_matrix(key_pose_0[:3, :3]).as_quat()

        # poses = [
        #     Pose(key_pose_0[:3, 3] + np.array([0.0, 0.0, 0.12]), Rotation.from_matrix(key_pose_0[:3, :3]).as_quat(scalar_first=True)),
        #     Pose(key_pose_0[:3, 3], Rotation.from_matrix(key_pose_0[:3, :3]).as_quat(scalar_first=True))
        # ]
        poses = [
            Pose(key_pose_0[:3, 3] + np.array([0.0, 0.0, 0.12]), np.array([key_pose_0_quat[3], key_pose_0_quat[0], key_pose_0_quat[1], key_pose_0_quat[2]])),
            Pose(key_pose_0[:3, 3], np.array([key_pose_0_quat[3], key_pose_0_quat[0], key_pose_0_quat[1], key_pose_0_quat[2]]))
        ]
        
        trajectory = robot_util.get_trajectory(init_qpos=INIT_QPOS, poses=poses, gripper_action=0)
        # print(trajectory)
        ee_pose_list.extend(trajectory[0])
        for i in range(len(trajectory[1])):
            qpos = trajectory[1][i]
            qpos[-2:] = sum(INIT_QPOS[-2:]) / 2
            qpos_list.append(qpos)
        action_list.extend(trajectory[2])
        key_ee_pose = ee_pose_list[-1]

        qpos = robot_util.robot.get_qpos()
        for i in range(num_grasp_step):
            qpos[-2:] = ((grasp_width - sum(INIT_QPOS[-2:])) / num_grasp_step * (i + 1) + sum(INIT_QPOS[-2:])) / 2
            ee_pose_list.append(robot_util.get_ee_pose())
            qpos_list.append(qpos)
            action = np.ones([8,])
            action[:7] = trajectory[2][-1][:7]
            action_list.append(action)
        
        
        key_pose_1 = np.eye(4)
        key_pose_1[:3, 3] = ee_pose_in_demo[key_frame_indices[1]][:3, 3] + displacement_pos
        rot_euler = (Rotation.from_euler('Z', displacement_rot) * Rotation.from_matrix(ee_pose_in_demo[key_frame_indices[1]][:3, :3])).as_euler('XYZ')
        rot_euler[2] = normalize_and_adjust_angle(rot_euler[2])
        key_pose_1[:3, :3] = Rotation.from_euler('XYZ', rot_euler).as_matrix()
        # poses = [
        #     Pose(key_pose_1[:3, 3], Rotation.from_matrix(key_pose_1[:3, :3]).as_quat(scalar_first=True)),
        # ]
        key_pose_1_quat = Rotation.from_matrix(key_pose_1[:3, :3]).as_quat()
        poses = [
            Pose(key_pose_1[:3, 3], np.array([key_pose_1_quat[3], key_pose_1_quat[0], key_pose_1_quat[1], key_pose_1_quat[2]])),
        ]
        trajectory = robot_util.get_trajectory(init_qpos=qpos, poses=poses, gripper_action=1)
        ee_pose_list.extend(trajectory[0])
        for i in range(len(trajectory[1])):
            qpos = trajectory[1][i]
            qpos[-2:] = grasp_width / 2
            qpos_list.append(qpos)
        action_list.extend(trajectory[2])

        object_gaussian_aug = copy.deepcopy(object_gaussian_origin)
        T_rot = get_T_for_rotating_around_an_axis(ee_pose_in_demo[key_frame_indices[0]][0, 3], ee_pose_in_demo[key_frame_indices[0]][1, 3], displacement_rot)
        T_rot = torch.from_numpy(T_rot).cuda().to(torch.float32)
        object_gaussian_aug = transform_gaussian(object_gaussian_aug, T_rot)
        actual_displacement_pos = key_ee_pose[:3] - ee_pose_in_demo[key_frame_indices[0]][:3, 3]
        actual_displacement_pos[2] = 0.0  # z should not be modified
        object_gaussian_aug._xyz = object_gaussian_aug._xyz + torch.from_numpy(actual_displacement_pos).cuda()

        # mango_gaussian_aug = copy.deepcopy(mango_gaussian)

        episode_length = len(qpos_list)
        image_0_list = []
        image_1_list = []
        image_2_list = []
        for i in range(episode_length):
            # object gaussian
            if i > 0 and action_list[i-1][-1] == 0 and action_list[i][-1] == 1:
                ref_ee_pose = ee_pose_list[i]
            if action_list[i][-1] == 1:
                ee_pose = ee_pose_list[i]
                delta_pos = ee_pose[:3] - ref_ee_pose[:3]
                object_gaussian = copy.deepcopy(object_gaussian_aug)
                object_gaussian._xyz = object_gaussian_aug._xyz + torch.from_numpy(delta_pos).cuda()
            else:
                object_gaussian = copy.deepcopy(object_gaussian_aug)
            # robot gaussian
            robot_gaussian = get_robot_gaussian_at_qpos(qpos_list[i])
            # print(qpos_list[i])
            # all gaussians


            gaussian_all = GaussianModel(sh_degree=3)
            # Trans = pose_to_transformation_matrix(ee_pose_list[i]) @ rotation_z(180)
            Trans = pose_to_transformation_matrix(ee_pose_list[i]) @ rotation_z(90)

            Trans[2,3] = Trans[2,3] + 0.05
            camera_2, renderer_2= get_changed_camera_and_renderer(Trans)
            # mango_gaussian_aug._xyz = mango_gaussian._xyz*0.01 + torch.from_numpy(Trans[:3,3]).cuda()
            



            if augment_appearance:
                gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right = get_gaussian_plane(texture=True)
                gaussian_all.compose([robot_gaussian, object_gaussian, gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right])
            elif augment_lighting:
                _, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right = get_gaussian_plane(texture=False)
                gaussian_all.compose([robot_gaussian, object_gaussian, table_gaussian,  gaussian_plane_front, gaussian_plane_left, gaussian_plane_right])
            else:
                gaussian_all.compose([robot_gaussian, object_gaussian, table_gaussian])
            gaussian_all._xyz = gaussian_all._xyz.to(torch.float32)

            # augment lighting
            if augment_lighting:
                gaussian_all = augment_lighting_for_scene(gaussian_all)
            # augment the camera view
            if augment_camera_pose:
                camera_0, renderer_0, camera_1, renderer_1 = get_augmented_camera_and_renderer()
            # print(pose_to_transformation_matrix(ee_pose_list[i]))


            rgb_0 = renderer_0.render(gaussian_all)
            rgb_0 = (np.clip(rgb_0.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            image_0_list.append(rgb_0)
            rgb_0 = np.array(Image.fromarray(rgb_0).resize([image_size, image_size], Image.ANTIALIAS))

            rgb_1 = renderer_1.render(gaussian_all)
            rgb_1 = (np.clip(rgb_1.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            image_1_list.append(rgb_1)
            rgb_1 = np.array(Image.fromarray(rgb_1).resize([image_size, image_size], Image.ANTIALIAS))

            rgb_2 = renderer_2.render(gaussian_all)
            rgb_2 = (np.clip(rgb_2.detach().cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
            image_2_list.append(rgb_2)
            rgb_2 = np.array(Image.fromarray(rgb_2).resize([image_size, image_size], Image.ANTIALIAS))

            data_recorder.append_step_in_current_episode({
                'qpos': qpos_list[i], 
                'ee_pose': ee_pose_list[i], 
                'image_0': rgb_0,
                'image_1': rgb_1,
                'action': action_list[i]
            })


        data_recorder.save_current_episode()

        if save_video:
            save_rgb_images_to_video(image_0_list, f'{output_path}/demo_{demo_idx}_0.mp4')
            save_rgb_images_to_video(image_1_list, f'{output_path}/demo_{demo_idx}_1.mp4')
            save_rgb_images_to_video(image_2_list, f'{output_path}/demo_{demo_idx}_2.mp4')


if __name__ == '__main__':
    main(ref_demo_path, xy_step_str, augment_lighting, augment_appearance, augment_camera_pose, output_path, image_size, save, save_video)
