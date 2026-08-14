import numpy as np

#Transforms camera coordinates to Franka coords
def franka_coordTransform(coords):

    #Camera to robot base transformation matrix
    T_CAM_TO_ROBOT = np.array([[ 0.0124, -0.9864, -0.1640,  0.4471],
                               [-0.9996, -0.0078, -0.0284, -0.4186],
                               [ 0.0267,  0.1643, -0.9860,  1.2320],
                               [ 0.0000,  0.0000,  0.0000,  1.0000]])