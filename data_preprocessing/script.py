import os, sys
sys.path.append('.')

import train_utils
from dataset.dataset_smpl import DatasetSMPL
from dataset.dataset_synthetic import DatasetSynthetic
import numpy as np

smpl_dir = '/data/yushuo/smpl_models'
FLAGS = train_utils.get_flags()
dataset_name = FLAGS.dataset_name

if os.path.isfile(os.path.join(FLAGS.data_dir, 'calibration_full.json')):
    with open(os.path.join(FLAGS.data_dir, 'calibration_full.json'), 'r') as fp:
        import json
        cam_data = json.load(fp)
    cam_ssn_list = sorted(list(cam_data.keys()))
    open(os.path.join(FLAGS.data_dir, 'cam_ssns.txt'), 'w').write('\n'.join(cam_ssn_list))
    
    if os.path.isfile(os.path.join(FLAGS.data_dir, 'whole.pt')):
        dataset_train    = DatasetSMPL(FLAGS.data_dir, None, FLAGS, validate=False)
    else:
        dataset_train    = DatasetSynthetic(FLAGS.data_dir, None, FLAGS, validate=False)

os.makedirs('/tmp', exist_ok=True)
np.save('/tmp/beta.npy', FLAGS.beta)
print("Beta: ", FLAGS.beta)
os.makedirs('data/data_templates/%s' % dataset_name, exist_ok=True)

print("python -m data_preprocessing.export_smpl_model %s %s" % (dataset_name, smpl_dir))
os.system("python -m data_preprocessing.export_smpl_model %s %s" % (dataset_name, smpl_dir))
print("python -m data_preprocessing.compute_diffused_skinning_smplx %s %s" % (dataset_name, smpl_dir))
os.system("python -m data_preprocessing.compute_diffused_skinning_smplx %s %s" % (dataset_name, smpl_dir))
print("python data_preprocessing/front_back_rast.py %s" % dataset_name)
os.system("python data_preprocessing/front_back_rast.py %s" % dataset_name)