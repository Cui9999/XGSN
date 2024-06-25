import numpy as np
import cv2
import os
import torch
from plyfile import PlyData
from PIL import Image
import open3d as o3d


# 假设点云数据为 point_cloud，形状为 (N, 3)，其中 N 表示点的数量
def generate_random_camera_poses(num_views):
    R = []
    T = []
    for _ in range(num_views):
        rvec, _ = cv2.Rodrigues(np.random.randn(3))
        R.append(rvec)
        T.append(np.random.randn(3))
    return R, T


def project_point_cloud_to_image_single_view(point_cloud, R, T, K, image_shape, view_index):

    point_cloud_rotated = np.matmul(point_cloud, R[view_index].T) + T[view_index]

    point_cloud_rotated_abs = np.abs(point_cloud_rotated)
    # Project the rotated point cloud to the image plane
    points_2d, _ = cv2.projectPoints(point_cloud_rotated_abs, np.zeros((3,)), np.zeros((3,)), K, None)

    # Normalize the projected points to image coordinates
    points_2d_norm = np.squeeze(points_2d).astype(int)
    points_2d_norm[:, 0] = np.clip(points_2d_norm[:, 0], 0, image_shape[1] - 1)
    points_2d_norm[:, 1] = np.clip(points_2d_norm[:, 1], 0, image_shape[0] - 1)

    # Create an empty image
    image = np.zeros(image_shape, dtype=np.uint8)

    # Compute the depth of each point and map it to the range [0, 255]
    for (x, y), point in zip(points_2d_norm, point_cloud_rotated_abs):
        z = point[2]
        depth = (z - np.min(point_cloud_rotated_abs[:, 2])) / (
                    np.max(point_cloud_rotated_abs[:, 2]) - np.min(point_cloud_rotated_abs[:, 2]))
        image[y, x] = depth * 255
    image = image[300:1100, 220:1020]
    image = np.resize(image, (400, 400))

    return image


def project_point_cloud_to_image(point_cloud, R, T, K, image_shape, num_views):
    # Use joblib Parallel to perform the projection in parallel
    projected_images = []
    for i in range(num_views):
        project_single_view = project_point_cloud_to_image_single_view(point_cloud, R, T, K, image_shape, i)
        projected_images.append(project_single_view)
        # testimage = Image.fromarray(project_single_view)
        # testimage.show(str(i))
        # testimage.save('E:\\Cui\\MANet-master\\data\\' + str(i) + '.png')

    projected_images = np.stack(projected_images, axis=0)
    projected_images = torch.from_numpy(projected_images)

    return projected_images


def read_ply_file(file_path):
    # Read the PLY file
    plydata = PlyData.read(file_path)

    # Get the data from the PLY file
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']

    # Convert the data to a NumPy array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud

#
# folder_path = 'E:\\Cui\\MANet-master\\data\\PointCloudNew'
# point_clouds = []
#
#
# # Sample camera intrinsics (replace with your actual camera parameters)
# K = np.array([[500, 0, 320],
#               [0, 500, 240],
#               [0, 0, 1]], dtype=np.float32)
# # Generate random camera poses
# num_views = 3
#
# R = np.load('E:\\Cui\\MANet-master\\data\\R.npy')
# T = np.load('E:\\Cui\\MANet-master\\data\\T.npy')
# # np.save('E:\\Cui\\MANet-master\\data\\R.npy', R)
# # np.save('E:\\Cui\\MANet-master\\data\\T.npy', T)
#
# # file_list = os.listdir(folder_path)
# # for i, filename in enumerate(file_list):
# #     if filename.endswith(".ply"):
# #         new_filename = f"{'bs_'}{str(i+1).zfill(3)}.ply"
# #         os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
# for file in os.listdir(folder_path):
#     if file.endswith('.ply'):
#         file_path = os.path.join(folder_path, file)
#         point_cloud = read_ply_file(file_path)
#         point_clouds.append(point_cloud)
#         projected_images = project_point_cloud_to_image(point_cloud, R, T, K, [1300, 1300], num_views)
#         for i, image in enumerate(projected_images):
#             file_name = f'_{i + 1}.png'
#             save_path = os.path.join('E:\\Cui\\MANet-master\\data\\点云多视角图', os.path.splitext(file)[0]+file_name)
#             # 将处理后的图像保存为PNG格式
#             image_pil = Image.fromarray(image)  # 将NumPy数组转换为PIL图像
#             image_pil = image_pil.crop((300, 220, 1100, 1020))
#             image_pil = image_pil.resize((400, 400))
#             # image_pil.show(str(i))
#             image_pil.save(save_path)

