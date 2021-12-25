import numpy as np
import random
from path import Path
import os
from joblib import Parallel, delayed


path = Path("data/ModelNet40")

folders = [dir for dir in sorted(os.listdir(path)) if os.path.isdir(path/dir)]
classes = {folder: i for i, folder in enumerate(folders)}

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

with open(path/"bed/train/bed_0001.off", 'r') as f:
  verts, faces = read_off(f)

i,j,k = np.array(faces).T
x,y,z = np.array(verts).T

class PointSampler(object):
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0) ** 0.5

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        # https://mathworld.wolfram.com/BarycentricCoordinates.html
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t - s) * pt2[i] + (1 - t) * pt3[i]
        return (f(0), f(1), f(2))

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros((len(faces)))

        for i in range(len(areas)):
            areas[i] = (self.triangle_area(verts[faces[i][0]],
                                           verts[faces[i][1]],
                                           verts[faces[i][2]]))

        sampled_faces = (random.choices(faces,
                                        weights=areas,
                                        cum_weights=None,
                                        k=self.output_size))

        sampled_points = np.zeros((self.output_size, 3))

        for i in range(len(sampled_faces)):
            sampled_points[i] = (self.sample_point(verts[sampled_faces[i][0]],
                                                   verts[sampled_faces[i][1]],
                                                   verts[sampled_faces[i][2]]))

        return sampled_points


pointcloud = PointSampler(10000)((verts, faces))



def process(file, file_adr, save_adr):
    fname = save_adr + '/' + file[:-4] + '.npy'
    if file_adr.endswith('.off'):
        if not os.path.isfile(fname):
            print(file)
            with open(file_adr, 'r') as f:
                try:
                    verts, faces = read_off(f)
                    pointcloud = PointSampler(10000)((verts, faces))
                    np.save(fname, pointcloud)
                except:
                    pass
        else:
            pass






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

for category in classes.keys():
    save_adr = 'data/ModelNet_40_npy/' + category + '/' + folder
    try:
        os.makedirs(save_adr)
    except:
        pass
    new_dir = root_dir/Path(category)/folder
    for file in os.listdir(new_dir):
        all_files.append(file)
        all_files_adr.append(new_dir/file)
        all_save_adr.append(save_adr)

Parallel(n_jobs=40)(delayed(process)(file, file_adr, save_adr) for (file, file_adr, save_adr)
                   in zip(all_files, all_files_adr, all_save_adr))

folder = 'test'
root_dir = path
folders = [dir for dir in sorted(os.listdir(root_dir)) if os.path.isdir(root_dir/dir)]
classes = {folder: i for i, folder in enumerate(folders)}
files = []

all_files= []
all_files_adr = []
all_save_adr = []

for category in classes.keys():
    save_adr = 'data/ModelNet_40_npy/' + category + '/' + folder
    try:
        os.makedirs(save_adr)
    except:
        pass
    new_dir = root_dir/Path(category)/folder
    for file in os.listdir(new_dir):
        all_files.append(file)
        all_files_adr.append(new_dir/file)
        all_save_adr.append(save_adr)

Parallel(n_jobs=40)(delayed(process)(file, file_adr, save_adr) for (file, file_adr, save_adr)
                   in zip(all_files, all_files_adr, all_save_adr))

