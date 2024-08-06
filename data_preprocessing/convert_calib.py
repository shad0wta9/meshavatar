import cv2
import os, sys
import tqdm
import numpy as np

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}

    camnames = []
    for key_ in camera.keys():
        camnames.append(key_)

    # camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        
        val['K'] = np.array(val['K']).reshape(3, 3)
        val['R'] = np.array(val['R']).reshape(3, 3)
        val['T'] = np.array(val['T'])
        val['distCoeff'] = np.array(val['distCoeff'])

        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['distCoeff'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])


data_dir = 'xxxxx/synthetic_human_pp/jody'
intri_name = data_dir + '/intri.yml'
extri_name = data_dir + '/extri.yml'
intri = FileStorage(intri_name)
extri = FileStorage(extri_name)

d = {}
for i in range(20):
    dd = {
        'K': np.asarray(intri.fs.getNode('K_%02d' % i).mat()).reshape(-1).tolist(),
        'R': np.asarray(extri.fs.getNode('Rot_%02d' % i).mat()).reshape(-1).tolist(),
        'T': np.asarray(extri.fs.getNode('T_%02d' % i).mat()).reshape(-1).tolist(),
        'distCoeff': np.asarray(intri.fs.getNode('dist_%02d' % i).mat()).reshape(-1).tolist(),
        'imgSize': [1024, 1024],
        "rectifyAlpha": 0.0
    }
    d['%02d' % i] = dd

import json
with open(data_dir + '/calibration_full.json', 'w') as f:
    json.dump(d, f)