from utils import get_weight_path,get_weight_list

__all__ = ['r3d_18', 'se_r3d_18','da_18','da_se_18','r3d_34','se_r3d_34','da_34','da_se_34','vgg16_3d','vgg19_3d']


NET_NAME = 'se_r3d_18'
VERSION = 'v2.0-maxpool'
DEVICE = '2'
# Must be True when pre-training and inference
PRE_TRAINED = False 
# 1,2,3,4,5
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5
TTA_TIMES = 5

CKPT_PATH = './ckpt/{}/fold{}'.format(VERSION,CURRENT_FOLD)
WEIGHT_PATH = get_weight_path(CKPT_PATH)
# print(WEIGHT_PATH)

if PRE_TRAINED:
    WEIGHT_PATH_LIST = get_weight_list('./ckpt/{}/'.format(VERSION))
else:
    WEIGHT_PATH_LIST = None

# Arguments when trainer initial
INIT_TRAINER = {
    'net_name':NET_NAME,
    'lr':1e-3, 
    'n_epoch':120,
    'channels':1,
    'num_classes':2,
    'input_shape':(256,128,128),
    'crop':0,
    'scale':None,
    'use_roi':False or 'roi' in VERSION,
    'batch_size':2,
    'num_workers':2,
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'weight_path':WEIGHT_PATH,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'use_fp16':True,
    'use_maxpool': True if 'maxpool' in VERSION else False
 }

# Arguments when perform the trainer 
SETUP_TRAINER = {
    'output_dir':'./ckpt/{}'.format(VERSION),
    'log_dir':'./log/{}'.format(VERSION),
    'optimizer':'AdamW',
    'loss_fun':'Cross_Entropy',
    'class_weight':[1.0,2.0],
    'lr_scheduler':'MultiStepLR', # MultiStepLR
    'repeat_factor':3.0
}

