import numpy as np

def pose_to_transformation_matrix(pose):
    # print(len(list(pose)))
    # if len(list(pose)) != 7:
    #     pose
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


def compare_arrays_diff(arr1, arr2):
    """
    返回两个大小相同 NumPy 数组对应元素差值平方之和

    Args:
        arr1 (numpy.ndarray): 第一个数组
        arr2 (numpy.ndarray): 第二个数组

    Returns:
        float: 差值平方之和

    Raises:
        ValueError: 如果两个数组大小不同
    """
    # 检查两个数组的大小是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("两个数组大小必须相同")

    # 计算对应元素的差值平方
    diff_sq = (arr1 - arr2) ** 2

    # 计算差值平方之和
    total_diff_sq_sum = np.sum(diff_sq)

    # 判断差值平方之和是否小于阈值
    return total_diff_sq_sum

def compare_arrays_diff_sq_sum_less_than(arr1, arr2, threshold):
    """
    比较两个大小相同 NumPy 数组对应元素差值平方之和是否小于给定阈值

    Args:
        arr1 (numpy.ndarray): 第一个数组
        arr2 (numpy.ndarray): 第二个数组
        threshold (float): 阈值

    Returns:
        bool: 差值平方之和小于阈值返回 True，否则返回 False

    Raises:
        ValueError: 如果两个数组大小不同
    """
    # 检查两个数组的大小是否相同
    if arr1.shape != arr2.shape:
        raise ValueError("两个数组大小必须相同")

    # 计算对应元素的差值平方
    diff_sq = (arr1 - arr2) ** 2

    # 计算差值平方之和
    total_diff_sq_sum = np.sum(diff_sq)

    # 判断差值平方之和是否小于阈值
    return total_diff_sq_sum < threshold