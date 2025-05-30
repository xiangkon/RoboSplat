import os
import shutil
import h5py
import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, InterpolationMode


class DataRecorder:
    def __init__(self, save, data_path, start_episode_idx=0, image_shape=[256, 256], image_channel_first=False, save_video=False):
        self.save = save
        self.data_path = data_path
        os.makedirs(self.data_path, exist_ok=True)
        self.save_video = save_video
        self.image_channel_first = image_channel_first
        self.image_processor = Compose([
            Resize(image_shape, interpolation=InterpolationMode.BICUBIC),
        ])
        self.current_episode = []
        self.episode_idx = start_episode_idx

    def append_step_in_current_episode(self, data_dict):
        # data_dict: {'qpos': qpos, 'ee_pose': ee_pose, 'image_0': image_0, 'image_1': image_1, 'action': action}
        # image: np.uint8, range from 0 to 255
        if not self.save:
            return
        self.current_episode.append(data_dict)

    def save_current_episode(self):
        if not self.save:
            return
        num_step = len(self.current_episode)
        # qpos
        qposes = np.array([step['qpos'] for step in self.current_episode])
        # ee_pose
        ee_poses = np.array([step['ee_pose'] for step in self.current_episode])
        # image
        images_0 = torch.from_numpy(np.array([step['image_0'] for step in self.current_episode]))
        if not self.image_channel_first:
            images_0 = images_0.permute(0, 3, 1, 2)
        images_0 = self.image_processor(images_0).permute(0, 2, 3, 1).numpy()
        # image 1
        images_1 = torch.from_numpy(np.array([step['image_1'] for step in self.current_episode]))
        if not self.image_channel_first:
            images_1 = images_1.permute(0, 3, 1, 2)
        images_1 = self.image_processor(images_1).permute(0, 2, 3, 1).numpy()
        # action
        actions = np.array([step['action'] for step in self.current_episode])
        # actions[:, :7] = actions[:, :7] - qposes[:, :7]
        
        assert qposes.shape[0] == ee_poses.shape[0] == images_0.shape[0] == actions.shape[0]

        with h5py.File(f'{self.data_path}/{str(self.episode_idx).zfill(6)}.h5', 'w') as h5_file:
            h5_file.create_dataset(name='num_step', data=qposes.shape[0])
            obs_group = h5_file.create_group(name='obs')
            obs_group.create_dataset(name='qpos', data=qposes)
            obs_group.create_dataset(name='ee_pose', data=ee_poses)
            obs_group.create_dataset(name='image_0', data=images_0)
            obs_group.create_dataset(name='image_1', data=images_1)
            h5_file.create_dataset(name='action', data=actions)
        print(f'save episode {self.episode_idx} in {self.data_path}')

        self.current_episode = []
        self.episode_idx += 1

    def clean(self):
        self.current_episode = []
    
def compile_statistic(save_path, source_dir, dataset_key):
    os.makedirs(save_path, exist_ok=True)
    filtered_data_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if dataset_key in f]
    total_num_step = 0
    total_num_episode = 0
    episode_end_idx = []
    for data_path in filtered_data_paths:
        for filename in os.listdir(data_path):
            if filename.startswith('meta') or filename.endswith('mp4'):
                continue
            print(total_num_episode, data_path, filename)
            try:
                h5_file = h5py.File(os.path.join(data_path, filename), 'r')
            except:
                continue
            action = h5_file['action'][()]
            qpos = h5_file['obs']['qpos'][()]
            ee_pose = h5_file['obs']['ee_pose'][()]
            num_step = len(action)
            total_num_step += num_step
            episode_end_idx.append(total_num_step)
            shutil.copy(os.path.join(data_path, filename), os.path.join(save_path, str(total_num_episode).zfill(6) + '.h5'))

            total_num_episode += 1
            if 'statistic_max' not in locals():
                statistic_max = {'action': np.max(action, axis=0), 'qpos': np.max(qpos, axis=0), 'ee_pose': np.max(ee_pose, axis=0)}
                statistic_min = {'action': np.min(action, axis=0), 'qpos': np.min(qpos, axis=0), 'ee_pose': np.min(ee_pose, axis=0)}
                statistic_sum = {'action': np.sum(action, axis=0), 'qpos': np.sum(qpos, axis=0), 'ee_pose': np.sum(ee_pose, axis=0)}
            else:
                statistic_max = {'action': np.maximum(statistic_max['action'], np.max(action, axis=0)), 'qpos': np.maximum(statistic_max['qpos'], np.max(qpos, axis=0)), 'ee_pose': np.maximum(statistic_max['ee_pose'], np.max(ee_pose, axis=0))}
                statistic_min = {'action': np.minimum(statistic_min['action'], np.min(action, axis=0)), 'qpos': np.minimum(statistic_min['qpos'], np.min(qpos, axis=0)), 'ee_pose': np.minimum(statistic_min['ee_pose'], np.min(ee_pose, axis=0))}
                statistic_sum = {'action': statistic_sum['action'] + np.sum(action, axis=0), 'qpos': statistic_sum['qpos'] + np.sum(qpos, axis=0), 'ee_pose': statistic_sum['ee_pose'] + np.sum(ee_pose, axis=0)}

    statistic_mean = {'action': statistic_sum['action'] / total_num_step, 'qpos': statistic_sum['qpos'] / total_num_step, 'ee_pose': statistic_sum['ee_pose'] / total_num_step}

    for data_path in filtered_data_paths:
        for filename in os.listdir(data_path):
            if filename.startswith('meta') or filename.endswith('mp4'):
                continue
            h5_file = h5py.File(os.path.join(data_path, filename), 'r')
            action = h5_file['action'][()]
            qpos = h5_file['obs']['qpos'][()]
            ee_pose = h5_file['obs']['ee_pose'][()]
            if 'statistic_sum_of_square' not in locals():
                statistic_sum_of_square = {'action': np.sum((action - statistic_mean['action']) ** 2, axis=0), 'qpos': np.sum((qpos - statistic_mean['qpos']) ** 2, axis=0), 'ee_pose': np.sum((ee_pose - statistic_mean['ee_pose']) ** 2, axis=0)}
            else:
                statistic_sum_of_square = {'action': statistic_sum_of_square['action'] + np.sum((action - statistic_mean['action']) ** 2, axis=0), 'qpos': statistic_sum_of_square['qpos'] + np.sum((qpos - statistic_mean['qpos']) ** 2, axis=0), 'ee_pose': statistic_sum_of_square['ee_pose'] + np.sum((ee_pose - statistic_mean['ee_pose']) ** 2, axis=0)}

    statistic_std = {'action': (statistic_sum_of_square['action'] / (total_num_step - 1)) ** 0.5, 'qpos': (statistic_sum_of_square['qpos'] / (total_num_step - 1)) ** 0.5, 'ee_pose': (statistic_sum_of_square['ee_pose'] / (total_num_step - 1)) ** 0.5}

    with h5py.File(f'{save_path}/meta.h5', 'w') as h5_file:
        h5_file.create_dataset(name='num_episode', data=total_num_episode)
        h5_file.create_dataset(name='num_step', data=total_num_step)
        h5_file.create_dataset(name='episode_end_idx', data=np.array(episode_end_idx))
        statistic_group = h5_file.create_group(name='statistic')
        statistic_action_group = statistic_group.create_group(name='action')
        statistic_action_group.create_dataset(name='min', data=statistic_min['action'])
        statistic_action_group.create_dataset(name='max', data=statistic_max['action'])
        statistic_action_group.create_dataset(name='mean', data=statistic_mean['action'])
        statistic_action_group.create_dataset(name='std', data=statistic_std['action'])
        statistic_qpos_group = statistic_group.create_group(name='qpos')
        statistic_qpos_group.create_dataset(name='min', data=statistic_min['qpos'])
        statistic_qpos_group.create_dataset(name='max', data=statistic_max['qpos'])
        statistic_qpos_group.create_dataset(name='mean', data=statistic_mean['qpos'])
        statistic_qpos_group.create_dataset(name='std', data=statistic_std['qpos'])
        statistic_ee_pose_group = statistic_group.create_group(name='ee_pose')
        statistic_ee_pose_group.create_dataset(name='min', data=statistic_min['ee_pose'])
        statistic_ee_pose_group.create_dataset(name='max', data=statistic_max['ee_pose'])
        statistic_ee_pose_group.create_dataset(name='mean', data=statistic_mean['ee_pose'])
        statistic_ee_pose_group.create_dataset(name='std', data=statistic_std['ee_pose'])
