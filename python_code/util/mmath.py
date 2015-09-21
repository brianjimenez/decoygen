__author__ = 'bjimenez@bsc.es'

import random
from math import cos, sin
import numpy as np


def get_random_rotation_matrix():
    """Calculates a random rotation matrix using theta x, y and z"""
    tx = 359*random.random()
    ty = 179*random.random()
    tz = 359*random.random()
    Rx = np.array([[1,0,0], [0, cos(tx), -sin(tx)], [0, sin(tx), cos(tx)]])
    Ry = np.array([[cos(ty), 0, -sin(ty)], [0, 1, 0], [sin(ty), 0, cos(ty)]])
    Rz = np.array([[cos(tz), -sin(tz), 0], [sin(tz), cos(tz), 0], [0,0,1]])
    return np.dot(Rx, np.dot(Ry, Rz)).flatten().reshape((3, 3))


def get_affine(m):
    b = np.zeros((4, 4))
    b[:3, :3] = m
    b[3,3] = 1.
    return b