import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt

from utils import timing_decorator


sigma = 1.0
step = 0.1
x_arr = np.arange(-5, 5, step)
y_arr = np.arange(-5, 5, step)


@timing_decorator
def serial_make_mexican_hat():
    xx, yy, zz = list(), list(), list()
    for x in x_arr:
        for y in y_arr:
            z = (1. / (np.pi * np.power(sigma, 4))) * (1.0 - 0.5 * (x**2 + y**2) / sigma**2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
            
            xx.append(x)
            yy.append(y)
            zz.append(z)
    
    grid_shape = (x_arr.shape[0], y_arr.shape[0])
    xx = np.array(xx).reshape(*grid_shape)
    yy = np.array(yy).reshape(*grid_shape)
    zz = np.array(zz).reshape(*grid_shape)
    return xx, yy, zz


@timing_decorator
def make_mexican_hat():
    yy, xx = np.meshgrid(y_arr, x_arr)
    zz = (1. / (np.pi * np.power(sigma, 4))) * (1.0 - 0.5 * (xx**2 + yy**2) / sigma**2) * np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return xx, yy, zz


if __name__ == '__main__':
    xx, yy, zz = serial_make_mexican_hat()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, zz)
    plt.show()

    _xx, _yy, _zz = make_mexican_hat()
    assert np.isclose(xx, _xx).all()
    assert np.isclose(yy, _yy).all()
    assert np.isclose(zz, _zz).all()

    '''
    more details on meshgrid:
    https://stackoverflow.com/questions/36013063/what-is-the-purpose-of-meshgrid-in-python-numpy
    '''
