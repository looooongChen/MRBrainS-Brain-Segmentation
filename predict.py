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

running_config = {
    'model_dir': './models/model_2013_nopp',
    'batch_size': 4
}

# setup for MRBrainS2013
dataset_config = {
    'data_dir': './MRBrainS2013/testData',
    'channels': {
        'flair': 'T2_FLAIR.nii',
        'ir': 'T1_IR.nii',
        't1': 'T1.nii'
    },
    'input_channels': ['t1', 'flair', 'ir']
}

# create dataset
dataset = Dataset(dataset_config)
dataset.read_subjects()

subjects = [str(num+1) for num in range(15)]

outputs = single_view_predict(net_config, running_config,
                              dataset, subjects, output_type='label')

# save the results
save_dir = './MRBrainS2013/results'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for index, subject in enumerate(outputs,1):
    subject = np.moveaxis(subject, 0, -1)
    img = nib.Nifti1Image(subject.astype(np.uint8), np.eye(4))
    img.to_filename(os.path.join(save_dir, 'Segm_'+subjects[index-1]+'.nii'))

# save in MRBrainS 2013 submission format
# save_dir = './2013/submission'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# for index, subject in enumerate(outputs,1):
#     subject = np.moveaxis(subject, 0, -1)
#     subject = translate_label_2013_test(subject)
#     img = nib.Nifti1Image(subject.astype(np.uint8), np.eye(4))
#     img.to_filename(os.path.join(save_dir, 'Segm_MRBrainS13_{:02d}.nii'.format(index)))
