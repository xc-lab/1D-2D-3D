import numpy as np
from scipy import interpolate



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


# Brightness
def brightness_shift(data, shift_max_value):
    color = data[:,4:7]
    m, n = color.shape
    matrix = np.random.uniform(-shift_max_value, shift_max_value, (m,n))
    color += matrix
    data[:,4:7] = color
    return data


def brightness_scaling(data, max_value):
    color = data[:, 4:7]
    m,n = color.shape
    matrix = np.random.uniform(1 - max_value, 1 + max_value, (m,n))
    color *= matrix
    data[:,4:7] = color
    return data

# jitter
def location_jittering(data, max_value):
    location = data[:, 0:2]
    m,n = location.shape
    matrix = np.random.uniform(1-max_value, 1+max_value, (m,n))
    location *= matrix
    data[:, 0:2] = location
    return data