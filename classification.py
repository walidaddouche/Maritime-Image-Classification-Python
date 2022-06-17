import numpy
from PIL import Image
from numpy import asarray
import os
import pandas as pd

# load the image
path_sea = "./Data/Mer"
path_other = "./Data/Ailleurs"
i = 0
import numpy as np

sea_list = []
others_list = []
images = []
import matplotlib.pyplot as plt

for img in os.listdir(path_sea):
    image = plt.imread(os.path.join(path_sea, img))
    sea_list.append(image)
    images.append(image)
for img in os.listdir(path_other):
    image = plt.imread(os.path.join(path_other, img))
    others_list.append(image)
    images.append(image)
sea_ndarray = np.array(sea_list, dtype=np.ndarray)
others_ndarray = np.array(others_list, dtype=np.ndarray)
X = np.array([sea_ndarray, others_ndarray], dtype=numpy.ndarray)
Y = np.array([[1 for i in range(207)], [(-1) for i in range(207)]], dtype=numpy.ndarray)

