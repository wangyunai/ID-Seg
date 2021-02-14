import os
import time
import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import nibabel as nb
from PIL import Image
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from sklearn.metrics import confusion_matrix, f1_score
import itertools


plane_view = "axial" 
# Pretrain Options
pretrained_num_classes = 88
# Provide data set path here.
image_dir = ''
label_dir = ['CSF labels dir', 'WM lables dir', 'etc']
num_class = 5
batch_size = 8
# number of subject in dataset
all_subj = list(range(1, 11)) 
device="cuda:2"
max_epochs = 16
# example: '3D_Models/axial_not_{}_{}.pth'
trained_model_path = ''
# example: '3D_predictions/axial_epoch_{}.npy'
predictions_save_path = ''

params = {'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_class': pretrained_num_classes,
        'se_block': False,
        'drop_out':0.2} 

class ProcessedDataset(data.Dataset):
    def __init__(self, label_dir, image_dir, selected_subjs, plane_view, load_in_ram=True):
        """
        Args:
            label_dir(string):path to the label image file
            image_dir(string):path to the original image file
            selected_subjs is the list of subjects included in the data set
        """
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.selected_subjs = selected_subjs
        self.plane_view = plane_view
        self.load_in_ram = load_in_ram
        temp_list = []
        for file in os.listdir(self.image_dir):
            if file.endswith('.nii.gz'):
                if int(file.split('_')[1][1:]) in self.selected_subjs:
                    temp_list.append(os.path.join(image_dir, file))
        temp_list = sorted(temp_list)
        if self.load_in_ram:
            self.img_lst = []
            for img_path in temp_list:
                self.img_lst.append(nb.load(img_path).get_data().astype(float))
        else:
            self.img_lst = temp_list
        self.label_lst = []
        for one_label_dir in self.label_dir:
            temp_list = []
            for file in os.listdir(one_label_dir):
                if file.endswith('.nii.gz'):
                    if int(file.split('_')[1][1:]) in self.selected_subjs:
                        temp_list.append(os.path.join(one_label_dir, file))
            temp_list = sorted(temp_list)
            if self.load_in_ram:
                one_label_list = []
                for label_path in temp_list:
                    one_label_list.append(nb.load(label_path).get_data().astype(float))
                self.label_lst.append(one_label_list)
            else:
                self.label_lst.append(temp_list)
        self.slice_cnt = self.number_of_slices_in_each_image()
        
    def number_of_slices_in_each_image(self):
        slice_cnt = []
        for img in self.img_lst:
            if self.load_in_ram:
                data = img
            else:
                data = nb.load(img).get_data()
            if self.plane_view == 'axial':
                slice_cnt.append(data.shape[2])
            elif self.plane_view == 'coronal':
                slice_cnt.append(data.shape[1])
            elif self.plane_view == 'sagittal':
                slice_cnt.append(data.shape[0]) 
            else:
                print('plane_view should be axial, coronal, or sagittal!!')
                sys.exit()
        return slice_cnt
    
    def __len__(self):
        return sum(self.slice_cnt)
    
    def find_idx(self, idx, slice_cnt):
        """
        idx is zero_based
        """
        result = 0
        for i in range(len(slice_cnt)):
            result += slice_cnt[i]
            if idx < result:
                return i, idx - (result - slice_cnt[i])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subj_id, slice_idx = self.find_idx(idx, self.slice_cnt)
        if self.load_in_ram:
            img = self.img_lst[subj_id]
        else:
            img = nb.load(self.img_lst[subj_id]).get_data().astype(float)
        if self.plane_view == 'axial':
            img = img[1:-1, 1:-1, slice_idx]
        elif self.plane_view == 'coronal':
            img = img[1:-1, slice_idx, 5:-6]
        elif self.plane_view == 'sagittal':
            img = img[slice_idx, 1:-1, 5:-6]
        img = torch.from_numpy(img)
        
        label = np.zeros_like(img)
        label_num = 1
        for one_label_lst in self.label_lst:
            if self.load_in_ram:
                one_label = one_label_lst[subj_id]
            else:
                one_label = nb.load(one_label_lst[subj_id]).get_data().astype(float)

            if self.plane_view == 'axial':
                one_label = one_label[1:-1, 1:-1, slice_idx]
            elif self.plane_view == 'coronal':
                one_label = one_label[1:-1, slice_idx, 5:-6]
            elif self.plane_view == 'sagittal':
                one_label = one_label[slice_idx, 1:-1, 5:-6] 
                   
            one_label = one_label[1:-1, 1:-1, slice_idx]
            label += one_label * label_num
            label_num += 1
        label = torch.from_numpy(label)
        if len(img.shape) == 2:
            img = torch.unsqueeze(img,0)
            label = torch.unsqueeze(label,0)
        sample = {'image': img, 'label': label}

        return sample


class Network(nn.Module):
    # define your onw architecture use conv block and up/down conv layer
    def __init__(self):
        
        super(Network, self).__init__()
        params['num_channels'] = 1
        params['num_class'] = pretrained_num_classes
        self.encode1 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode3 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.encode4 = sm.EncoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.bottleneck = sm.DenseBlock(params, se_block_type=se.SELayer.CSSE)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode2 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode3 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        self.decode4 = sm.DecoderBlock(params, se_block_type=se.SELayer.CSSE)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)
        
    def forward(self, x):
        e1, out1, ind1 = self.encode1.forward(x)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        e4, out4, ind4 = self.encode4.forward(e3)
        
        bn = self.bottleneck.forward(e4)

        d4 = self.decode4.forward(bn, out4, ind4)
        d3 = self.decode1.forward(d4, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)

        return d1
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.network = Network()
        params['num_class'] = num_class
        self.classifier = sm.ClassifierBlock(params)
        
        
    def forward(self, x):
        x = self.network.forward(x)
        x = self.classifier.forward(x)
        
        return x
        

for epoch in range(1, max_epochs):
    subj_preds_axial = []
    for test_subj in all_subj:
        print('test subject: {}'.format(test_subj))
        model = Classifier()
        model_path = trained_model_path.format(test_subj, epoch)
        model.load_state_dict(torch.load(model_path, map_location="cuda:2"))
        model.eval()
        model.to(device)

        dataset=ProcessedDataset(label_dir, image_dir, [test_subj], plane_view)
        dataloader=data.DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

        total_voxels = 0
        correct_voxels = 0
        preds = []
        for i,d in enumerate(dataloader):
            img,label=d['image'].to(device,dtype=torch.float),d['label']
            label = label.numpy()[:, 0, :, :]
            out = model(img)
            out = out.cpu().detach().numpy()
            predictions = np.argmax(out, axis = 1)
            correct_voxels += np.sum(label == predictions)
            total_voxels += label.shape[0] * label.shape[1] * label.shape[2]
            preds.append(out)

        subj_preds_axial.append(np.concatenate(preds))
        print(float(correct_voxels) / total_voxels)

    with open(predictions_save_path.format(epoch), 'wb') as f:
        for pred in subj_preds_axial:
            np.save(f, pred)
            