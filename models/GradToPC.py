import cv2
import numpy as np
import os
from plyfile import PlyData, PlyElement
from PIL import Image


def depth_to_point_cloud(depth_image):
    # Step 1: Get depth image dimensions
    width = depth_image.shape[0]
    height = depth_image.shape[1]
    # Step 2: Initialize empty point cloud
    point_cloud = []

    # Step 3: Convert depth image to point cloud
    for v in range(height):
        for u in range(width):
            depth = depth_image[v, u]
            if 10 < depth < 240:
                # Step 3.1: Convert depth to 3D coordinates
                z = depth
                x = u
                y = v

                # Step 3.2: Append 3D point to point cloud
                point_cloud.append([x, y, z])

    return np.array(point_cloud)


# Folder path containing images
folder_path = 'E:\\Cui\\MANet-master\\data\\点云灰度图'
folder_path2 = 'C:\\Users\\Cui\\Desktop\\所有点云'
# Get a list of all image files in the folder
image_files = [f for f in os.listdir(folder_path2) if f.endswith(('.jpg', '.png', '.bmp'))]


# Process each image and convert it to a point cloud
for image_file in image_files:
    # Load the image using PIL
    image_pil = Image.open(os.path.join(folder_path2, image_file))
    image_pil = image_pil.resize((224, 224))
    # Load the grayscale image (replace 'path_to_image' with the path to your image)
    gray_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2GRAY)

    # Assuming you already have the depth_image from the previous step
    # Convert depth image to point cloud
    point_cloud = depth_to_point_cloud(gray_image)

    # Create PlyData object
    vertices = np.array([(x, y, z) for x, y, z in point_cloud], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_element = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([vertex_element])

    # Save the point cloud to a .ply file
    output_filename = os.path.splitext(image_file)[0] + '.ply'
    output_path = os.path.join('E:\\Cui\\MANet-master\\data', output_filename)
    plydata.write(output_path)


