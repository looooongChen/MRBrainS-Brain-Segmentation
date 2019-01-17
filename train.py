from preprocess import *
from model import *
from estimator import *
import os


tf.logging.set_verbosity(tf.logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# create config object

net_config = {
    'type': 'combi',
    # 9 lables for MRBrainS2013
    'class_num': 9,
    # 11 labels for MRBrainS2018
    # 'class_num': 11,
    'stream_num': 3,
    'with_pp': False,
    'filter_num': 24,
    'subsample': 2,
    'block_num': (3, 2, 3, 2, 3),
    'block_factor': 2,
    'block_dilation': (2, 4, 8, 16, 32)
}

training_config = {
    'model_dir': './models_2013/model_all_nopp',
    'epoch': 250,
    'batch_size': 4,
    'init_lr': 1e-4,
    'weight_decay': 1e-3,
    'momentum': 0.99
}

# MRBrainS2013 dataset config
dataset_config = {
    'data_dir': './MRBrainS2013/trainingData',
    'channels': {
        'flair': 'T2_FLAIR.nii',
        'ir': 'T1_IR.nii',
        't1': 'T1.nii',
        'seg': 'LabelsForTraining.nii'
    },
    'input_channels': ['t1', 'flair', 'ir'],
    'output_channels': ['seg']
}

# MRBrainS2018 dataset config
# dataset_config = {
#     'data_dir': './MRBrainS2018/trainingData',
#     'channels': {
#         'flair': 'pre/FLAIR.nii.gz',
#         'ir': 'pre/reg_IR.nii.gz',
#         't1': 'pre/reg_T1.nii.gz',
#         'seg': 'segm.nii.gz'
#     },
#     'input_channels': ['t1', 'flair', 'ir'],
#     'output_channels': ['seg']
# }

# create dataset
dataset = Dataset(dataset_config)
dataset.read_subjects()
# subjects of MRBrainS2013
train_subjects = ['1', '2', '3', '4', '5']
# subjects of MRBrainS2018
# train_subjects = ['1', '4', '5', '7', '14', '70', '148']
val_subjects = ['1']
train_np = dataset.generate_ds(train_subjects, True)
val_np = dataset.generate_ds(val_subjects, False)

# start training
train(net_config, training_config, train_np, val_np)



