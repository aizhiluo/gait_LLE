import numpy as np
import matplotlib.pyplot as plt

def trans(key):
    if key == 'heart':
        return heartTransform
    if key == 'bow':
        return bowTransform
    if key == 'acrobat':
        return acrobatTransform

def heartTransform(t):
    return 1.8*np.sin(t)**3, 1.3*np.cos(t)-0.5*np.cos(2*t)-0.2*np.cos(3*t)-    \
                0.1*np.cos(4*t)

def bowTransform(t):
    return 1.4*np.cos(t), 2.9*np.sin(t)*np.cos(t)

def acrobatTransform(t):
    return 1.6*np.cos(0.5*t)*(np.sin(0.5*t+0.5)+np.cos(1.5*t+0.5)),            \
            1.6*np.sin(0.5*t)*(np.sin(0.5*t+0.5)-np.cos(1.5*t+0.5))