import os
import copy
import math
import torch
from torch import nn
from torch import einsum
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
import numpy as np
from e3nn import o3
import einops
from plyfile import PlyData, PlyElement
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_aug.util.robot_util import FrankaEmikaPanda


def rotate_around_z(vector, rot):
    Rz = np.array([
        [np.cos(rot), -np.sin(rot), 0],
        [np.sin(rot),  np.cos(rot), 0],
        [0,            0,           1]
    ])
    v_rotated = Rz @ vector

    return v_rotated

def get_T_for_rotating_around_an_axis(x, y, angle):
    T1 = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    Rz = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    T2 = np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    M = T2 @ Rz @ T1

    return M

def get_T_for_rotating_around_an_axis_along_x(y, z, angle):
    T1 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0, 1]
    ])
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(angle), -np.sin(angle), 0],
        [0, np.sin(angle),  np.cos(angle), 0],
        [0, 0, 0, 1]
    ])
    T2 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    M = T2 @ Rx @ T1

    return M

def get_T_for_rotating_around_an_axis_along_y(x, z, angle):
    T1 = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, 0],
        [0, 0, 1, -z],
        [0, 0, 0, 1]
    ])
    Ry = np.array([
        [ np.cos(angle), 0, np.sin(angle), 0],
        [ 0, 1, 0, 0],
        [-np.sin(angle), 0, np.cos(angle), 0],
        [ 0, 0, 0, 1]
    ])
    T2 = np.array([
        [1, 0, 0, x],
        [0, 1, 0, 0],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])
    M = T2 @ Ry @ T1

    return M

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device='cuda')
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    L = R @ L

    return L

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)

    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P

def skew_symmetric(w):
    w0,w1,w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O,-w2,w1],dim=-1),
                        torch.stack([w2,O,-w0],dim=-1),
                        torch.stack([-w1,w0,O],dim=-1)],dim=-2)
    return wx

def taylor_A(x,nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i>0: denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans
    
def taylor_B(x,nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def taylor_C(x,nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans

def se3_to_SE3(w,v): # [...,3]
    deltaT = torch.zeros((4,4)).cuda()
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)
    I = torch.eye(3,device=w.device,dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    deltaT[:3, :3] = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    deltaT[:3, 3] = V@v
    deltaT[3, 3] = 1.
    return deltaT

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5

class GaussianModel:
    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device='cuda')
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def construct_list_of_attributes(self, ignore_f_rest=False):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if not ignore_f_rest:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, ignore_f_rest=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(ignore_f_rest)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if ignore_f_rest:
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        else:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]['x']),
                        np.asarray(plydata.elements[0]['y']),
                        np.asarray(plydata.elements[0]['z'])),  axis=1)
        opacities = np.asarray(plydata.elements[0]['opacity'])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]['f_dc_0'])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]['f_dc_1'])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]['f_dc_2'])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('f_rest_')]
        no_f_rest = False
        if len(extra_f_names) == 0:
            no_f_rest = True
            for i in range(3*(self.max_sh_degree + 1) ** 2 - 3):
                extra_f_names.append('f_rest_{}'.format(i))
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            if no_f_rest:
                features_extra[:, idx] = np.zeros([xyz.shape[0],])
            else:
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device='cuda'), requires_grad=False)
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=False)
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device='cuda').transpose(1, 2).contiguous(), requires_grad=False)
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device='cuda'), requires_grad=False)
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device='cuda'), requires_grad=False)
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device='cuda'), requires_grad=False)

        self.active_sh_degree = self.max_sh_degree

    def compose(self, gaussian_model_list):
        self._xyz = nn.Parameter(torch.cat([gm._xyz for gm in gaussian_model_list], dim=0))
        self._features_dc = nn.Parameter(torch.cat([gm._features_dc for gm in gaussian_model_list], dim=0))
        self._features_rest = nn.Parameter(torch.cat([gm._features_rest for gm in gaussian_model_list], dim=0))
        self._opacity = nn.Parameter(torch.cat([gm._opacity for gm in gaussian_model_list], dim=0))
        self._scaling = nn.Parameter(torch.cat([gm._scaling for gm in gaussian_model_list], dim=0))
        self._rotation = nn.Parameter(torch.cat([gm._rotation for gm in gaussian_model_list], dim=0))

class RealCamera(nn.Module):
    def __init__(self, R, T, c2w, fovy, fovx, znear=0.1, zfar=10.0, image_size=[1600, 1600]):
        super().__init__()
        self.image_width = image_size[1]
        self.image_height = image_size[0]
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        if c2w is None:
            start_pose_w2c = torch.tensor(getWorld2View2(R, T)).cuda()
        else:
            start_pose_w2c = torch.from_numpy(np.linalg.inv(c2w)).cuda().to(torch.float32)
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).cuda())
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).cuda())

        self.forward(start_pose_w2c)

    def forward(self, start_pose_w2c):
        deltaT = se3_to_SE3(self.w, self.v)
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()

    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class Camera(nn.Module):
    def __init__(self, R, T, fovy, znear=0.1, zfar=10.0, image_size=[1600, 1600]):
        super().__init__()
        self.image_width = image_size[1]
        self.image_height = image_size[0]
        self.FoVy = np.deg2rad(fovy)
        self.FoVx = 2 * np.arctan(np.tan(self.FoVy / 2) * (self.image_width / self.image_height))
        self.znear = znear
        self.zfar = zfar

        start_pose_w2c = torch.tensor(getWorld2View2(R, T)).cuda()
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).cuda())
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).cuda())

        self.forward(start_pose_w2c)

    def forward(self, start_pose_w2c):
        deltaT = se3_to_SE3(self.w, self.v)
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()

    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def render_rgb(camera, gaussian, bg_color):
    raster_settings = GaussianRasterizationSettings(
        image_height=int(camera.image_height),
        image_width=int(camera.image_width),
        tanfovx=math.tan(camera.FoVx * 0.5),
        tanfovy=math.tan(camera.FoVy * 0.5),
        bg=torch.tensor(bg_color).to(torch.float32).cuda(),
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=gaussian.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        antialiasing=True
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    rendered_image, radii, depth_image = rasterizer(
        means3D=gaussian.get_xyz,
        means2D=torch.zeros_like(gaussian.get_xyz, dtype=gaussian.get_xyz.dtype, requires_grad=False, device="cuda"),
        shs=gaussian.get_features,
        opacities=gaussian.get_opacity,
        scales=gaussian.get_scaling,
        rotations=gaussian.get_rotation,
    )
    rendered_image = rendered_image.clamp(0, 1)

    return rendered_image.permute(1, 2, 0)

class GaussianRenderer():
    def __init__(self, camera, bg_color=[0.0, 0.0, 0.0]):
        self.camera = camera
        self.is_camera_list = False
        if isinstance(self.camera, list):
            self.is_camera_list = True
            self.num_camera = len(self.camera)
            self.cnt = 0
        self.bg_color = bg_color
    
    def render(self, gaussian):
        if self.is_camera_list:
            idx = self.cnt % (2 * self.num_camera)
            if idx >= self.num_camera:
                idx = 2 * self.num_camera - (idx + 1)
            rgba = render_rgb(self.camera[idx], gaussian, self.bg_color)
            self.cnt += 1
            return rgba
        else:
            return render_rgb(self.camera, gaussian, self.bg_color)

def transform_shs(shs_feat, rotation_matrix):
    ## rotate shs
    P = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).cuda() # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    #rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_1,
        one_degree_shs,  
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_2,
        two_degree_shs,
    )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
        "... i j, ... j -> ... i",
        D_3,
        three_degree_shs,
    )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat

def transform_gaussian(gaussian, T, scale=1.0, transform_sh=False):
    # transform xyz
    xyz = gaussian.get_xyz
    xyz = xyz * scale
    xyz = xyz @ T[:3, :3].T + T[:3, 3]
    gaussian._xyz = xyz
    # tranform rotation
    rotation = gaussian.get_rotation
    rotation = quaternion_to_matrix(rotation)
    rotation = T[:3, :3] @ rotation
    rotation = matrix_to_quaternion(rotation)
    gaussian._rotation = rotation
    # transform scale
    gaussian._scaling = torch.log(torch.exp(gaussian._scaling) * scale)

    if transform_sh:
        # transform sh
        sh = gaussian._features_rest
        sh = transform_shs(sh, T[:3, :3])
        gaussian._features_rest = sh

    return gaussian

def invert_transformation_matrix(T):
    batch_size = T.shape[0]
    rotation = T[:, :3, :3]  # Shape: (batch_size, 3, 3)
    translation = T[:, :3, 3]  # Shape: (batch_size, 3)
    inv_rotation = rotation.transpose(1, 2)  # Transpose each rotation matrix
    inv_translation = -torch.bmm(inv_rotation, translation.unsqueeze(2)).squeeze(2)
    inv_T = torch.eye(4).repeat(batch_size, 1, 1).to(T.device)
    inv_T[:, :3, :3] = inv_rotation
    inv_T[:, :3, 3] = inv_translation

    return inv_T

LINK2SEG = [11, 12, 13, 14, 15, 16, 17, 18, 21, 22]  # 10 parts mapping to actor segmentation masks in sapien images
FIXED_LINK = ['panda_link0', 'panda_link8', 'panda_hand', 'panda_hand_tcp', 'panda_leftfinger_pad', 'panda_rightfinger_pad']  # Those links are fixed and should be skipped
robot = FrankaEmikaPanda()
robot_part_gaussian_origin_list = []

def get_robot_gaussian_at_qpos(qpos, return_gripper=False):
    if len(robot_part_gaussian_origin_list) == 0:
        for i in range(10):
            robot_part_gaussian = GaussianModel(sh_degree=3)
            robot_part_gaussian.load_ply(f'data/gaussian/robot/link{i}_default.ply')
            robot_part_gaussian_origin_list.append(robot_part_gaussian)

    T_all = robot.fkine_all(qpos)
    T_dof = []  # Transformation at current pose
    for i in range(len(robot.links)):
        if robot.links[i].name in FIXED_LINK:
            continue
        T_dof.append(T_all[i+1])  # skip the base link

    # apply the transformation on each part
    robot_part_gaussian_list = copy.deepcopy(robot_part_gaussian_origin_list)
    for i in range(1, 10):
        T_dof[i-1] = torch.from_numpy(T_dof[i-1].A).cuda().to(torch.float32)
        robot_part_gaussian_list[i] = transform_gaussian(robot_part_gaussian_list[i], T_dof[i-1])

    robot_gaussian = GaussianModel(sh_degree=3)
    robot_gaussian.compose(robot_part_gaussian_list)

    if return_gripper:
        gripper_gaussian = GaussianModel(sh_degree=3)
        gripper_gaussian.compose(robot_part_gaussian_list[-2:])
        return robot_gaussian, gripper_gaussian

    return robot_gaussian
