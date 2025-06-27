import os
import sapien
import numpy as np
import torch
import pytorch3d.ops as pytorch3d_ops
import mplib
import roboticstoolbox as rtb
from roboticstoolbox.robot.ERobot import ERobot


# INIT_QPOS = [0.0, 0.1, 0.0, -2.15, 0.0, 2.25, 0.78, 0.04, 0.04]
INIT_QPOS = [0.0, -0.6, 0.0, -2.3, 0.0, 1.9, 0.78, 0.04, 0.04]

class FrankaEmikaPanda(ERobot):
    '''
    Class that imports a FrankaEmikaPanda Robot
    '''
    def __init__(self):
        origin_dir = os.getcwd()
        links, name, urdf_string, urdf_filepath = super().URDF_read(
            rtb.rtb_path_to_datafile(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/asset/robot/panda/panda.urdf')), tld=FrankaEmikaPanda.load_my_path())
        os.chdir(origin_dir)
        super().__init__(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath)
        self.default_joint_pos = np.array(INIT_QPOS)
        self.addconfiguration('qr', self.default_joint_pos)

    @staticmethod
    def load_my_path():
        os.chdir(os.path.dirname(__file__))

class RobotUtil:
    def __init__(self, control_hz=22, timestep=1/180, image_size=[256, 256]):
        self.control_hz = control_hz
        frame_skip = int(1/timestep/control_hz)
        self.frame_skip = frame_skip
        self.setup_scene(image_size=image_size, timestep=timestep)
        self.load_robot()
        self.setup_planner()
        self.scene.update_render()

    def setup_scene(self, image_size, timestep, ray_tracing=True):
        self.scene = sapien.Scene()
        self.scene.set_timestep(timestep)
        self.scene.default_physical_material = self.scene.create_physical_material(1, 1, 0)
        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

        self.setup_camera(image_size)
        if ray_tracing:
            sapien.render.set_camera_shader_dir('rt')
            sapien.render.set_viewer_shader_dir('rt')
            sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser('oidn') # change to 'optix' or 'oidn'

    def load_robot(self, urdf_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/asset/robot/panda/panda.urdf')):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot = loader.load(urdf_path)
        self.robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.active_joints = self.robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(stiffness=1000, damping=100, force_limit=100, mode='force')
            joint.set_friction(0.0)
        self.init_qpos = INIT_QPOS
        self.robot.set_qpos(self.init_qpos)
        self.end_effector = self.robot.get_links()[10]
        for link in self.robot.links:
            link.disable_gravity = True

    def setup_planner(self, urdf_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/asset/robot/panda/panda.urdf'), srdf_path=os.path.join(os.path.abspath(os.path.dirname(__file__)), '../../data/asset/robot/panda/panda.srdf'), move_group='panda_hand_tcp'):
        self.planner = mplib.Planner(urdf=urdf_path, srdf=srdf_path, move_group=move_group)

    def setup_camera(self, image_size, fov=40, near=0.1, far=10.0):
        width = image_size[0]
        height = image_size[1]
        self.cameras = []

        cam_pos = np.array([1.4, 1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_0 = self.scene.add_camera(
            name='camera_0',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_0.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_0)

        cam_pos = np.array([1.4, -1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_1 = self.scene.add_camera(
            name='camera_1',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_1.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_1)

        cam_pos = np.array([-0.6, -1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_2 = self.scene.add_camera(
            name='camera_2',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_2.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_2)

        cam_pos = np.array([-0.6, 1.0, 1.4])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_3 = self.scene.add_camera(
            name='camera_3',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_3.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_3)

        cam_pos = np.array([1.4, 1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_4 = self.scene.add_camera(
            name='camera_4',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_4.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_4)

        cam_pos = np.array([1.4, -1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_5 = self.scene.add_camera(
            name='camera_5',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_5.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_5)

        cam_pos = np.array([-0.6, -1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_6 = self.scene.add_camera(
            name='camera_6',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_6.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_6)

        cam_pos = np.array([-0.6, 1.0, -0.6])
        look_at_point = np.array([0.4, 0.0, 0.4])
        cam_rel_pos = cam_pos - look_at_point
        forward = -cam_rel_pos / np.linalg.norm(cam_rel_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        camera_7 = self.scene.add_camera(
            name='camera_7',
            width=width,
            height=height,
            fovy=np.deg2rad(fov),
            near=near,
            far=far,
        )
        camera_7.entity.set_pose(sapien.Pose(mat44))
        self.cameras.append(camera_7)

    def get_ee_pose(self):
        ee_pose = self.end_effector.get_pose()
        return np.concatenate([ee_pose.p, ee_pose.q])

    def get_qpos(self):
        return self.robot.get_qpos()

    def get_pc(self, num_point=None):
        pc_list = []
        for camera in self.cameras:
            camera.take_picture()
            rgb = camera.get_picture('Color')[:,:,:3]
            position = camera.get_picture('Position')

            # segment robot
            seg_labels = camera.get_picture('Segmentation')  # [H, W, 4]
            label_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
            points_opengl = position[..., :3][(position[..., 3] < 1) & (label_image > 0)]  # position[..., :3][(position[..., 3] < 1)]
            points_color = rgb[(position[..., 3] < 1) & (label_image > 0)]

            model_matrix = camera.get_model_matrix()
            points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
            points_color = np.clip(points_color, 0, 1)  # (np.clip(points_color, 0, 1) * 255).astype(np.uint8)
            pc_list.append(np.concatenate([points_world, points_color], axis=-1))

        pc = np.concatenate(pc_list, axis=0)

        if num_point is not None:
            _, fps_idx = pytorch3d_ops.sample_farthest_points(points=torch.from_numpy(pc[:,:3]).cuda().unsqueeze(0), K=num_point)
            pc = pc[fps_idx[0].cpu().numpy()]

        return pc

    def get_pc_at_qpos(self, qpos, num_point=None):
        self.robot.set_qpos(qpos)
        self.scene.update_render()
        pc = self.get_pc(num_point)

        return pc

    def step(self, action):
        # position control
        for i in range(7):
            self.active_joints[i].set_drive_target(action[i])
        if action[-1] > 0.5:
            self.active_joints[-1].set_drive_target(0.00)
            self.active_joints[-2].set_drive_target(0.00)
        elif action[-1] <= 0.5:
            self.active_joints[-1].set_drive_target(0.04)
            self.active_joints[-2].set_drive_target(0.04)

        for _ in range(self.frame_skip):
            self.scene.step()
        self.scene.update_render()

    def get_trajectory(self, init_qpos, poses, gripper_action, control_hz=None):
        self.robot.set_qpos(init_qpos)
        ee_pose_list, qpos_list, action_list, n_step_list = [], [], [], [] # ee_pose and qpos are obs, i.e., they are the states before taking action

        for pose_idx, target_pose in enumerate(poses):
            result = self.planner.plan_pose(
                target_pose,
                self.get_qpos(),
                time_step=1/self.control_hz if control_hz is None else 1/control_hz[pose_idx],
            )
            # print(result)
            result_pos = result['position']
            # combine the small action
            # delta_action = abs(result_pos[1:] - result_pos[:-1])
            # delta_action.sum(axis=-1)
            result_pos[-3] = result_pos[-1]
            result_pos = result_pos[2:-2]

            n_step = result_pos.shape[0]
            for i in range(n_step):
                ee_pose_list.append(self.get_ee_pose())
                qpos_list.append(self.get_qpos())
                if gripper_action == 0:
                    action = np.zeros([8,])
                elif gripper_action == 1:
                    action = np.ones([8,])
                action[:7] = result_pos[i]
                action_list.append(action)
                self.step(action)
            n_step_list.append(n_step)

        return ee_pose_list, qpos_list, action_list, n_step_list
