import os
import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode, ToTensor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_aug.util.gaussian_util import *
from data_aug.util.camera_util import *


def normalize_and_adjust_angle(angle):
    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    
    return angle

def load_gaussian_plane():
    gaussian_plane_table = GaussianModel(sh_degree=3)
    gaussian_plane_table.load_ply('data/gaussian/black_plane_table.ply')
    gaussian_plane_table._xyz[:, 2] = (gaussian_plane_table._xyz[:, 0] - 0.5) * 0.02
    gaussian_plane_table._xyz[:, 2] -= 0.02
    gaussian_plane_front = GaussianModel(sh_degree=3)
    gaussian_plane_front.load_ply('data/gaussian/black_plane_front.ply')
    gaussian_plane_left = GaussianModel(sh_degree=3)
    gaussian_plane_left.load_ply('data/gaussian/black_plane_left.ply')
    gaussian_plane_right = GaussianModel(sh_degree=3)
    gaussian_plane_right.load_ply('data/gaussian/black_plane_right.ply')

    return gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right

class ToCudaTransform:
    def __init__(self):
        pass
    def __call__(self, image):
        return image.cuda()

def texture_gaussian_plane(gaussian_plane, height, width, image_path):
    # the points in gaussian_plane must be organized in the form of '[height, width, ...] -> [height * width, ...]'
    assert height * width == gaussian_plane._xyz.shape[0]
    image_processor = Compose([
        ToTensor(),
        ToCudaTransform(),
        Resize([height, width], interpolation=InterpolationMode.BICUBIC)
    ])
    texture = image_processor(Image.open(image_path)).permute(1, 2, 0)
    if texture.shape[2] == 1:  # gray
        texture = torch.cat([texture,]*3, dim=-1)
    features_dc = RGB2SH(texture[:,:,:3]).reshape(-1, 1, 3)
    gaussian_plane._features_dc = nn.Parameter(features_dc.cuda(),requires_grad=False)

    return gaussian_plane

gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right = load_gaussian_plane()

def get_gaussian_plane(texture=False):
    global gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right
    if texture:
        image_dir = 'data/coco_train2017_subset'
        num_image = 20
        gaussian_plane_table = texture_gaussian_plane(gaussian_plane_table, 725, 600, f'{image_dir}/{str(np.random.randint(0, num_image)).zfill(6)}.jpg')
        gaussian_plane_front = texture_gaussian_plane(gaussian_plane_front, 750, 750, f'{image_dir}/{str(np.random.randint(0, num_image)).zfill(6)}.jpg')
        gaussian_plane_left = texture_gaussian_plane(gaussian_plane_left, 750, 825, f'{image_dir}/{str(np.random.randint(0, num_image)).zfill(6)}.jpg')
        gaussian_plane_right = texture_gaussian_plane(gaussian_plane_right, 750, 825, f'{image_dir}/{str(np.random.randint(0, num_image)).zfill(6)}.jpg')

    return gaussian_plane_table, gaussian_plane_front, gaussian_plane_left, gaussian_plane_right

def augment_lighting_for_scene(gaussian, scale_range_min=0.3, scale_range_max=1.8, offset_range_min=-0.3, offset_range_max=0.3):
    gaussian._features_dc = SH2RGB(gaussian._features_dc)

    scale = torch.tensor([np.random.uniform(scale_range_min, scale_range_max), np.random.uniform(scale_range_min, scale_range_max), np.random.uniform(scale_range_min, scale_range_max)]).cuda().to(torch.float32)
    
    offset = torch.tensor([np.random.uniform(offset_range_min, offset_range_max), np.random.uniform(offset_range_min, offset_range_max), np.random.uniform(offset_range_min, offset_range_max)]).cuda().to(torch.float32)

    noise = torch.normal(mean=0, std=0.1, size=gaussian._features_dc.shape).cuda().to(torch.float32)

    gaussian._features_dc = torch.clamp(scale * gaussian._features_dc, min=0.0, max=1.0)
    gaussian._features_dc = torch.clamp(gaussian._features_dc + offset, min=0.0, max=1.0)
    gaussian._features_dc = torch.clamp(gaussian._features_dc + noise, min=0.0, max=1.0)
    gaussian._features_dc = RGB2SH(gaussian._features_dc)

    return gaussian

def get_camera_and_renderer():
    c2w = np.loadtxt('data/camera_pose/front_cam_pose.txt')
    c2w_1 = np.loadtxt('data/camera_pose/side_cam_pose.txt')

    camera_0 = RealCamera(R=None, T=None, c2w=c2w, fovy=0.7515759938811762, fovx=0.9684658025776031, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_0 = GaussianRenderer(camera_0, bg_color=[0.0, 0.0, 0.0])
    camera_1 = RealCamera(R=None, T=None, c2w=c2w_1, fovy=0.7552235860675748, fovx=0.9709616850130633, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_1 = GaussianRenderer(camera_1, bg_color=[0.0, 0.0, 0.0])

    return camera_0, renderer_0, camera_1, renderer_1


def get_camera_and_renderer_2():
    c2w = np.loadtxt('data/camera_pose/front_cam_pose_2.txt')
    c2w_1 = np.loadtxt('data/camera_pose/side_cam_pose_2.txt')

    camera_0 = RealCamera(R=None, T=None, c2w=c2w, fovy=0.7515759938811762, fovx=0.9684658025776031, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_0 = GaussianRenderer(camera_0, bg_color=[0.0, 0.0, 0.0])
    camera_1 = RealCamera(R=None, T=None, c2w=c2w_1, fovy=0.7552235860675748, fovx=0.9709616850130633, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_1 = GaussianRenderer(camera_1, bg_color=[0.0, 0.0, 0.0])

    return camera_0, renderer_0, camera_1, renderer_1
def get_camera_param():
    c2w = np.loadtxt('data/camera_pose/front_cam_pose.txt')
    c2w_1 = np.loadtxt('data/camera_pose/side_cam_pose.txt')
    r, theta, phi, r_1, theta_1, phi_1, look_at_point = camera_R_and_T_to_r_theta_phi_and_lookat_pos(c2w, c2w_1)

    return r, theta, phi, r_1, theta_1, phi_1, look_at_point

r, theta, phi, r_1, theta_1, phi_1, look_at_point = get_camera_param()
def get_augmented_camera_and_renderer():
    camera_pose_range_scale = 1.0
    look_at_point_range = camera_pose_range_scale * 0.1  # 0.1
    r_range = camera_pose_range_scale * 0.2  # 0.2
    theta_range = camera_pose_range_scale * np.pi / 6  # np.pi / 6
    phi_range = camera_pose_range_scale * np.pi / 6  # np.pi / 6
    yaw_range = camera_pose_range_scale * np.pi / 6  # np.pi / 6
    look_at_point_range1 = camera_pose_range_scale * 0.05
    r_range1 = camera_pose_range_scale * 0.08
    theta_range1 = camera_pose_range_scale * np.pi / 6 / 4  # np.pi / 6
    phi_range1 = camera_pose_range_scale * np.pi / 6 / 2 # np.pi / 6
    yaw_range1 = camera_pose_range_scale * np.pi / 6 / 2 # np.pi / 6
    curr_look_at_point = look_at_point + np.array([np.random.uniform(-look_at_point_range, look_at_point_range), np.random.uniform(-look_at_point_range, look_at_point_range), np.random.uniform(-look_at_point_range, look_at_point_range)])
    curr_look_at_point1 = look_at_point + np.array([np.random.uniform(-look_at_point_range1, look_at_point_range1), np.random.uniform(-look_at_point_range1, look_at_point_range1), np.random.uniform(-look_at_point_range1, look_at_point_range1)])
    curr_r = r + np.random.uniform(-r_range, r_range)
    curr_r_1 = r_1 + np.random.uniform(-r_range1, r_range1)
    curr_theta = theta + np.random.uniform(-theta_range, theta_range)
    curr_theta_1 = theta_1 + np.random.uniform(-theta_range1, theta_range1)
    curr_phi = phi + np.random.uniform(-phi_range, phi_range)
    curr_phi_1 = phi_1 + np.random.uniform(-phi_range1, phi_range1)
    R_0, T_0 = r_theta_phi_and_lookat_pos_to_camera_R_and_T(curr_r, curr_theta, curr_phi, curr_look_at_point, yaw=np.random.uniform(-yaw_range, yaw_range))
    R_1, T_1 = r_theta_phi_and_lookat_pos_to_camera_R_and_T(curr_r_1, curr_theta_1, curr_phi_1, curr_look_at_point1, yaw=np.random.uniform(-yaw_range1, yaw_range1))

    camera_0 = RealCamera(R=R_0, T=T_0, c2w=None, fovy=0.7515759938811762, fovx=0.9684658025776031, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_0 = GaussianRenderer(camera_0, bg_color=[0.0, 0.0, 0.0])
    camera_1 = RealCamera(R=R_1, T=T_1, c2w=None, fovy=0.7552235860675748, fovx=0.9709616850130633, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer_1 = GaussianRenderer(camera_1, bg_color=[0.0, 0.0, 0.0])

    return camera_0, renderer_0, camera_1, renderer_1

def get_changed_camera_and_renderer(c2w):

    # R_0, T_0 = get_R_T_fromw2c(c2w)

    # camera = RealCamera(R=R_0, T=T_0, c2w=None, fovy=0.7515759938811762, fovx=0.9684658025776031, znear=0.1, zfar=10.0, image_size=[480, 640])
    camera = RealCamera(R=None, T=None, c2w=c2w, fovy=0.7515759938811762, fovx=0.9684658025776031, znear=0.1, zfar=10.0, image_size=[480, 640])
    renderer = GaussianRenderer(camera, bg_color=[0.0, 0.0, 0.0])

    return camera, renderer