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

plane_view = "axial" 
# Pretrain Options
pretrained_model_path = ''
pretrained_num_classes = 88
# Provide data set path here.
image_dir = ''
label_dir = ['CSF labels dir', 'WM lables dir', 'etc']
num_class = 5
batch_size = 8
# number of subject in dataset
all_subj = list(range(1, 11)) 
lr_list = [1e-7, 1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 1e-2]

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
        
    
class Model(object):
    def __init__(self, batch, epoch, parallel, image_dir, label_dir, selected_subjs, pretrained_model_path, last_layer_trained_path, plane_view, workers=4, lr=5e-4, last_layer_only=True, device="cuda:0"):
        """
        Args:
            worker: number of workers for dataloader
            batch: batch size
            num_epochs: number of training epochs
            lr: learning rate
            img_dir: image dir for dataloader
            dataset: image dataset
            dataloader: image dataloader
            device: GPU
            ngpu: number of GPU
            last_layer_only: If it is true, the model will train just the last layer using the previous weights, and if it is   False, the model will train all the networks
            last_layer_trained_path: for trianing the whole network, it is better to first train the last layer only and then train the all layers from this network ( onece we fix all the layers and trian the last layer just in one epoch, and then tain all networks)
        """
        # self.logger=Logger('C:\\Users\\zhang\\Downloads\\Jupyter\\ISBI\\logs')
        # self.logger.writer.flush()

        self.workers = workers
        self.batch = batch
        self.num_epochs = epoch
        self.lr = lr
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.selected_subjs = selected_subjs
        self.pretrained_model_path = pretrained_model_path
        self.last_layer_trained_path = last_layer_trained_path

        self.dataset = ProcessedDataset(self.label_dir, self.image_dir, self.selected_subjs, plane_view)
        self.dataloader = data.DataLoader(self.dataset,batch_size=self.batch,
                                        shuffle=True,num_workers=self.workers)
        self.device = device
        self.parallel = parallel
        self.model = Classifier() 
        if last_layer_only:
            self.model.network.load_state_dict(torch.load(pretrained_model_path))
        else:
            self.model.load_state_dict(torch.load(self.last_layer_trained_path)) 
            
        if self.parallel and torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(device)
        if last_layer_only:
            self.optim = optim.Adam(self.model.classifier.parameters(), lr=self.lr, betas=(0.5, 0.999)) 
            for param in self.model.network.parameters():
                param.requires_grad = False
        else:        
            self.optim = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
            
    def L1_loss(self,inp,tar):
        return F.l1_loss(inp,tar)

    def cross_entropy_loss(self, inp, tar):
        loss = nn.CrossEntropyLoss()
        return loss(inp.reshape((inp.shape[0], params['num_class'], -1)), tar[:, 0, :, :].reshape(tar.shape[0], -1))

    def gen_loss(self,img,tar):
        return self.cross_entropy_loss(img, tar)
    # + self.L1_loss(img,tar.to(self.device,dtype=torch.float))

    def train_step(self,inp,tar):
        self.model.zero_grad()
        gen_opt = self.model(inp)

        g_loss = self.gen_loss(gen_opt, tar.to(self.device,dtype=torch.long))
        g_loss.backward() ### 
        self.optim.step()

        return g_loss.item()    


    def train(self):
        trainloader = self.dataloader
        device = self.device
        G_loss = []
        iters = 0
        print('start training:')
        for epoch in range (self.num_epochs):
            for i,data in enumerate(trainloader):
                img,label = data['image'].to(device,dtype=torch.float),data['label'].to(device,dtype=torch.long)
                g = self.train_step(img,label)
                iters += 1
                G_loss.append(g)
                if i%20 == 0:
                    print('[%d/%d][%d/%d]\tLoss_G: %.4f'
                      %(epoch,self.num_epochs,i,len(trainloader),G_loss[-1]))            
            if (epoch+1)%1 == 0:
        
                torch.save(self.model.state_dict(), save_path+str(epoch+1)+'.pth')
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': G_loss,
                }, '{}ckpt_'.format(save_path)+str(epoch+1)+'.pth')
    

        print('training finished!')   


        return self.model
    
start_time = time.time() 
###################################### If it is for training the last layer only, do this:######################################
#### - set last_layer_trained_path to None
#### - set last_layer_only to True
#### - Change save_path to something you like, for example '3D_Models/last_layer_trained_axial_not_{}_'
#### - Change the max_epoch to forexample 3
###################################### If traiing the whole model: #########################################################
#### - set last_layer_trained_path to the last_layer_trained model you want to use
#### - set last_layer_only to False
#### - Change save_path to something you like, for example '3D_Models/axial_not_{}_'
#### - Change the max_epoch to for example 15
##################################################################################################################

max_epoch = 15
device="cuda:0"
for lr in lr_list:
    for test_subj in all_subj: 
        last_layer_trained_path = '3D_Models/last_layer_trained_axial_not_{}_3.pth'.format(test_subj)
        save_path = '3D_Models/axial_not_{}_'.format(test_subj)
        selected_subjs = [x for x in all_subj if x not in [test_subj]]
        model = Model(batch_size, max_epoch, False, image_dir, label_dir, selected_subjs, pretrained_model_path, last_layer_trained_path, plane_view, last_layer_only=False, device=device, lr=lr)
        model.train()
        print(lr, test_subj, "--- {} minutes ---".format((time.time() - start_time)/ 60.0))
        
               
