#  -*- coding: utf-8 -*-
'''
augmentation methods
'''
import numpy as np
from scipy import interpolate
import open3d as o3d
import matplotlib.pyplot as plt
from decimal import Decimal


def normalization_array_data(data, data_range):
    normalized_data = data.copy()
    for i in np.arange(data.shape[1]):
        min_value, max_value = data_range[:, i]
        normalized_data[:,i] = (normalized_data[:,i]-min_value)/(max_value-min_value)
    return normalized_data


def get_column_data_scale(data):
    min_scale = np.min(data, axis=0)
    max_scale = np.max(data, axis=0)
    data_scale = np.vstack((min_scale, max_scale))
    return data_scale


def up_sampling_transform(data, scale):
    m, n = data.shape
    new_data = np.zeros((scale * m, n))
    for i in np.arange(n):
        x = np.linspace(0, m, m)
        y = data[:, i]
        xnew = np.linspace(0, len(data), scale * len(data))
        f = interpolate.interp1d(x, y, kind='linear')
        ynew = f(xnew)
        new_data[:, i] = ynew
    return new_data


def down_sampling_transform(data, scale):
    down_sampling_data = data[::scale]
    return down_sampling_data





# rotation
def rotation_transform(data, angle):
    data_range = get_column_data_scale(data)
    data = normalization_array_data(data, data_range)

    a_arr = np.array(data[:, 4])
    l_arr = np.array(data[:, 5])
    p_arr = np.array(data[:, 6])
    x_arr = np.array(data[:, 1])
    y_arr = np.array(data[:, 0])
    v_arr = np.array(data[:, 7])
    acc_arr = np.array(data[:, 8])
    jerk_arr = np.array(data[:, 9])

    # cmap = plt.cm.get_cmap('viridis')
    # color_cmap = cmap(np.arange(cmap.N))
    # colors = np.ones((acc_arr.shape[0], 3)).astype(np.float)
    # for i in np.arange(acc_arr.shape[0]):
    #     weight = acc_arr[i]
    #     new_weight = int(Decimal(255 * weight).quantize(Decimal("1."), rounding="ROUND_HALF_UP"))
    #     colors[i, :] = color_cmap[new_weight, :3]

    points = np.array([x_arr, y_arr, v_arr]).reshape(3, -1).T
    points_color = np.array([a_arr, l_arr, p_arr]).reshape(3, -1).T

    # points_color = colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack((points[:, 0], points[:, 1], points[:, 2])).transpose())
    pcd.colors = o3d.utility.Vector3dVector(
        np.vstack((points_color[:, 0], points_color[:, 1], points_color[:, 2])).transpose())

    R = pcd.get_rotation_matrix_from_xyz((0, 0, angle))  # 绕y轴旋转90°
    pcd = pcd.rotate(R)

    # o3d.visualization.draw_geometries([pcd], window_name='Raw Point Cloud data', mesh_show_wireframe=True,
    #                                   mesh_show_back_face=True)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1/126)
    voxels = voxel_grid.get_voxels()
    coordianate_indices = np.stack(list(vx.grid_index for vx in voxels))
    x_length = np.max(coordianate_indices.T[0])
    y_width = np.max(coordianate_indices.T[1])
    z_height = np.max(coordianate_indices.T[2])
    ex_array = np.ones((x_length + 2, y_width + 2, z_height + 2, 3)).astype(np.float64)
    for i in range(0, len(voxels)):
        ex_array[voxels[i].grid_index[0] + 1,
        voxels[i].grid_index[1] + 1,
        voxels[i].grid_index[2] + 1, :] = voxels[i].color

    return ex_array



# Brightness
def brightness_shift(data, shift_max_value):
    color = data[:,-3:]
    # color = np.reshape(color, (len(color), -1))
    m, n = color.shape
    matrix = np.random.uniform(-shift_max_value, shift_max_value, (m,n))
    color += matrix
    data[:,-3:] = color
    return data


def brightness_scaling(data, max_value):
    color = data[:, -3:]
    # color = np.reshape(color, (len(color), -1))
    m,n = color.shape
    matrix = np.random.uniform(1 - max_value, 1 + max_value, (m,n))
    color *= matrix
    # data[:,-3:] = color.ravel()
    data[:,-3:] = color
    return data

# jitter
def location_jittering(data, max_value):
    location = data[:, :2]
    m,n = location.shape
    matrix = np.random.uniform(-max_value, max_value, (m,n))
    location += matrix
    data[:,:2] = location[:,:]
    return data




