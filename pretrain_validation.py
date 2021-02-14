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

from PIL import Image
from nn_common_modules import modules as sm
from squeeze_and_excitation import squeeze_and_excitation as se
from scipy.special import softmax
from sklearn.metrics import f1_score

test_label_dir = ''
test_image_dir = ''
device = "cuda:1"
# pretrained model path for each plane view
axial_model_path = ''
coronal_model_path = ''
sagittal_model_path = ''
result_csv_path = ''
weight_axial = 0.4
weight_coronal = 0.4
weight_sagittal = 0.2

params = {'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'kernel_c':1,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_class':88,
        'se_block': False,
        'drop_out':0.2} 

class ProcessedDataset(data.Dataset):
    def __init__(self, label_dir, image_dir, plane_view, load_in_ram=False):
        """
        Args:
            label_dir(string):path to the label image file
            image_dir(string):path to the original image file
        """
        self.label_dir=label_dir
        self.image_dir=image_dir
        self.plane_view=plane_view
        temp_list=[]
        self.load_in_ram = load_in_ram
        for file in os.listdir(self.image_dir):
            if file.endswith('.nii.gz'):
                temp_list.append(os.path.join(image_dir, file))
        temp_list = sorted(temp_list)
        if self.load_in_ram:
            self.img_lst = []
            for img_path in temp_list:
                self.img_lst.append(nb.load(img_path).get_data().astype(float))
        else:
            self.img_lst = temp_list
        
        temp_list=[]
        for file in os.listdir(self.label_dir):
            if file.endswith('.nii.gz'):
                temp_list.append(os.path.join(label_dir, file))
        temp_list = sorted(temp_list)
        if self.load_in_ram:
            self.label_lst = []
            for lbl_path in temp_list:
                self.label_lst.append(nb.load(lbl_path).get_data().astype(float))
        else:
            self.label_lst = temp_list
            
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
            if len(slice_cnt) % 10 == 0:
                print("Processed {}/{}".format(len(slice_cnt), len(self.img_lst)))
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
        
        if self.load_in_ram:
            label = self.label_lst[subj_id]
        else:
            label = nb.load(self.label_lst[subj_id]).get_data().astype(float)
        if self.plane_view == 'axial':
            img = img[1:-1, 1:-1, slice_idx]
            label = label[1:-1, 1:-1, slice_idx] 
        elif self.plane_view == 'coronal':
            img = img[1:-1, slice_idx, 5:-6]
            label = label[1:-1, slice_idx, 5:-6]
        elif self.plane_view == 'sagittal':
            img = img[slice_idx, 1:-1, 5:-6 ]
            label = label[slice_idx, 1:-1, 5:-6 ]
               
        img = torch.from_numpy(img)
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
        prob = self.classifier.forward(d1)

        return prob


model = Network()

for batch_num in range(19):
    model.load_state_dict(torch.load(axial_model_path)) # overwrite the other model!
    model.eval()
    model.to(device)

    subj_preds_axial = []
    subj_true_label = []
    subj_images = sorted(os.listdir(test_image_dir))[5 * batch_num: 5 * (batch_num + 1)]
    subj_labels = sorted(os.listdir(test_label_dir))[5 * batch_num: 5 * (batch_num + 1)]
    for i in range(len(subj_images)):
        print(i)
        dataset=ProcessedDataset(os.path.join(test_label_dir, subj_labels[i]), os.path.join(test_image_dir, subj_images[i]), 'axial')
        dataloader=data.DataLoader(dataset,batch_size=8,num_workers=4, shuffle=False)
        start_time = time.time()
        preds = []
        y_test = []
        for i,d in enumerate(dataloader):
            img,label=d['image'].to(device,dtype=torch.float),d['label']
            label = label.numpy()[:, 0, :, :]
            out = model(img)
            out = out.cpu().detach().numpy()
            preds.append(out)
            y_test.append(label)
        subj_preds_axial.append(np.concatenate(preds))
        subj_true_label.append(np.concatenate(y_test))
        print("--- {} seconds ---".format((time.time() - start_time)))

    model.load_state_dict(torch.load(coronal_model_path)) # overwrite the other model!
    model.eval()
    model.to(device)

    subj_preds_coronal = []
    for i in range(len(subj_images)):
        print(i)
        dataset=ProcessedDataset(os.path.join(test_label_dir, subj_labels[i]), os.path.join(test_image_dir, subj_images[i]), 'coronal')
        dataloader=data.DataLoader(dataset,batch_size=8,num_workers=4, shuffle=False)
        start_time = time.time()
        preds = []
        for i,d in enumerate(dataloader):
            img,label=d['image'].to(device,dtype=torch.float),d['label']
            label = label.numpy()[:, 0, :, :]
            out = model(img)
            out = out.cpu().detach().numpy()
            preds.append(out)
        subj_preds_coronal.append(np.concatenate(preds))
        print("--- {} seconds ---".format((time.time() - start_time)))

    model.load_state_dict(torch.load(sagittal_model_path)) # overwrite the other model!
    model.eval()
    model.to(device)

    subj_preds_sagittal = []
    for i in range(len(subj_images)):
        print(i)
        dataset=ProcessedDataset(os.path.join(test_label_dir, subj_labels[i]), os.path.join(test_image_dir, subj_images[i]), 'sagittal')
        dataloader=data.DataLoader(dataset,batch_size=8,num_workers=4, shuffle=False)
        start_time = time.time()
        preds = []
        for i,d in enumerate(dataloader):
            img,label=d['image'].to(device,dtype=torch.float),d['label']
            label = label.numpy()[:, 0, :, :]
            out = model(img)
            out = out.cpu().detach().numpy()
            preds.append(out)
        subj_preds_sagittal.append(np.concatenate(preds))
        print("--- {} seconds ---".format((time.time() - start_time)))

    dice_scores = []
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
        dice_scores.append(tmp_dice_score)

    with open(result_csv_path, 'a') as f:
        for idx, dice_score in enumerate(dice_scores):
            f.write(subj_images[idx])
            f.write(",")
            f.write(",".join(map(str, dice_score)))
            f.write('\n')
