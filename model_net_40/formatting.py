import numpy as np
import random
import math
from path import Path
import os
#import plotly.graph_objects as go
from joblib import Parallel, delayed
import h5py


path = Path("data/ModelNet_40_npy")

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}

tr_label = []
tr_cloud = []
test_cloud = []
test_label = []

folder = 'train'
root_dir = path
folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
classes = {folder: i for i, folder in enumerate(folders)}
files = []

all_files= []
all_files_adr = []
all_save_adr = []

for (category, num) in zip(classes.keys(), classes.values()):
    new_dir = root_dir/Path(category)/folder

    for file in os.listdir(new_dir):
        print(file)
        if file.endswith('.npy'):
            try:
                point_cloud = np.load(new_dir / file)
                tr_cloud.append(point_cloud)
                tr_label.append(num)
            except:
                pass

tr_cloud = np.asarray(tr_cloud)
tr_label = np.asarray(tr_label)
np.save('data/tr_cloud.npy', tr_cloud)
np.save('data/tr_label.npy', tr_label)


folder = 'test'
root_dir = path
folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
classes = {folder: i for i, folder in enumerate(folders)}
files = []

for (category, num) in zip(classes.keys(), classes.values()):
    new_dir = root_dir/Path(category)/folder

    for file in os.listdir(new_dir):
        print(file)
        if file.endswith('.npy'):
            try:
                point_cloud = np.load(new_dir / file)
                test_cloud.append(point_cloud)
                test_label.append(num)
            except:
                pass

test_cloud = np.asarray(test_cloud)
test_label = np.asarray(test_label)
np.save('data/test_cloud.npy', test_cloud)
np.save('data/test_label.npy', test_label)

with h5py.File('data/ModelNet40_cloud.h5', 'w') as f:
    f.create_dataset("test_cloud", data=test_cloud)
    f.create_dataset("tr_cloud", data=tr_cloud)
    f.create_dataset("test_label", data=test_label)
    f.create_dataset("tr_label", data=tr_label)

'''f['test_cloud'] = test_cloud
f['tr_cloud'] = tr_cloud
f['test_label'] = test_label
f['tr_label'] = tr_label'''