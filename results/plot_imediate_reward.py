# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 13:39:50 2018

@author: yj-wn
"""
import glob
import numpy as np
import re
from matplotlib import pyplot as plt


def extract_number(p):
    return int(re.findall(r'\d+', p)[0])


def reward_plot(r):
    plt.figure(figsize=(12, 3))
    plt.plot(r)
    plt.show()


def reward_cum_plot(r):
    plt.figure(figsize=(12, 3))
    r = np.cumsum(r)
    plt.plot(r)
    plt.show()


paths = glob.glob('hist/*.npy')

# ordering paths
idx = np.array([extract_number(x) for x in paths])
paths = np.array(paths)
paths = paths[np.argsort(idx)]

# plotting reward
for i in range(len(paths)):
    r = np.load(paths[i])
    reward_plot(r)
    reward_cum_plot(r)
    print(i, ': ', np.sum(r))


for i in range(1700, 1720):
    r = np.load(paths[i])
    reward_plot(r)
    reward_cum_plot(r)
    print(i, ': ', np.sum(r))

############################
print(i)
i = 63
r = np.load(paths[i])
reward_plot(r)
np.sum(r)
#####
print(i)
i = 10
r = np.load(paths[i])
reward_plot(r)
np.sum(r)
#####
print(i)
i = 503
r = np.load(paths[i])
reward_plot(r)
np.sum(r)

