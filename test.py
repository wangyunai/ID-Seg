import os
import time
import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.nn as nn
import nibabel as nb
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools

from PIL import Image
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, f1_score
    
label_dir = ['CSF labels dir', 'WM lables dir', 'etc']
label_names = ['Background','CSF','WM','Hipoocampous','Amygdala']

# Provide saved prediction paths, for example '3D_predictions/axial_epoch_{}.npy'
axial_predictions_path = ''
coronal_predictions_path = ''
sagittal_predictions_path = ''

accuracy_csv_path = ''
dice_score_csv_path = ''

weight_axial = 0.4
weight_coronal = 0.4
weight_sagittal = 0.2

label_lst = []
for one_label_dir in label_dir:
    temp_list = []
    for file in os.listdir(one_label_dir):
        if file.endswith('.nii.gz'):
            temp_list.append(os.path.join(one_label_dir, file))
    temp_list = sorted(temp_list)
    one_label_list = []
    for label_path in temp_list:
        one_label_list.append(nb.load(label_path).get_data().astype(float))
    label_lst.append(one_label_list)

print('Labels loaded successfully!')

tmp_subj_true_label = []
for i in range(10):
    tmp_subj_true_label.append(np.zeros(label_lst[0][0][1:-1, 1:-1, :].shape, dtype=float))

label_num = 1
for one_label_lst in label_lst:
    for subj_id in range(10):
        one_label = one_label_lst[subj_id]
        one_label = one_label[1:-1, 1:-1, :]
        tmp_subj_true_label[subj_id] += one_label * label_num
    label_num += 1


subj_true_label = []
for label in tmp_subj_true_label:
    subj_true_label.append(np.swapaxes(np.swapaxes(label, 0, 1), 0, 2))

accuracy_result = []
dice_scores = []
for epoch in range(1, 16):
    print("======================================")
    print("Epoch: {}".format(epoch))
    
    subj_preds_axial = []
    with open(axial_predictions_path.format(epoch), 'rb') as f:
        for i in range(10):
            tmp = np.load(f)
            subj_preds_axial.append(tmp)

    subj_preds_coronal = []
    with open(coronal_predictions_path.format(epoch), 'rb') as f:
        for i in range(10):
            tmp = np.load(f)
            subj_preds_coronal.append(tmp)

    subj_preds_sagittal = []
    with open(sagittal_predictions_path.format(epoch), 'rb') as f:
        for i in range(10):
            tmp = np.load(f)
            subj_preds_sagittal.append(tmp)
            
    dice_score = np.zeros((5,))
    cnf_matrix = np.zeros((5,5))
    for i in range(len(subj_true_label)):
        print(i)
        true_label = subj_true_label[i][5:-6, :, :].flatten()
        axial = softmax(subj_preds_axial[i][5:-6, :, :], axis=1)
        coronal = softmax(subj_preds_coronal[i][1:-1, :, :], axis=1)
        coronal = np.swapaxes(coronal, 0, 3)
        sagittal = softmax(subj_preds_sagittal[i][1:-1, :, :], axis=1)
        sagittal = np.swapaxes(np.swapaxes(sagittal, 0, 2), 0, 3)
        final_pred = np.argmax(weight_axial * axial + weight_coronal * coronal + weight_sagittal * sagittal, axis=1).flatten()

        tmp_dice_score = f1_score(true_label, final_pred, average=None)
        print("Dice score: {}".format(tmp_dice_score))
        dice_score += tmp_dice_score
        cnf_matrix += confusion_matrix(true_label, final_pred)
        
    cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
    accuracy_result.append((epoch, cm.diagonal()))
    dice_scores.append((epoch, dice_score / len(subj_true_label)))

print("Start writing accuracies")
with open(accuracy_csv_path, 'w') as f:
    f.write('epoch,' + ','.join(label_names) + '\n')
    for el in accuracy_result:
        f.write("{},{},{},{},{},{}\n".format(el[0], el[1][0], el[1][1], el[1][2], el[1][3], el[1][4]))

print("Start writing dice scores")
with open(dice_score_csv_path, 'w') as f:
    f.write('epoch,' + ','.join(label_names) + '\n')
    for el in dice_scores:
        f.write("{},{},{},{},{},{}\n".format(el[0], el[1][0], el[1][1], el[1][2], el[1][3], el[1][4]))