import os
import time
import sys
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

# plane_view can be either axial, coronal, or sagittal
plane_view = "axial" 
save_path = '' + '_' + plane_view
image_path = ''
label_path = ''
batch_size = 8
epoch = 20

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

class Model(object):
    def __init__(self, batch, epoch, parallel, image_dir, label_dir, plane_view, workers=4, lr=5e-4, device="cuda:2"):
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
        """

        self.workers = workers
        self.batch = batch
        self.num_epochs = epoch
        self.lr = lr
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.dataset = ProcessedDataset(self.label_dir,self.image_dir, plane_view)
        self.dataloader = data.DataLoader(self.dataset,batch_size=self.batch,
                                        shuffle=True,num_workers=self.workers)
        self.device = device
        self.parallel = parallel
        self.network = Network()
        if self.parallel and torch.cuda.device_count()>1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.network = nn.DataParallel(self.network)

        self.network = self.network.to(device)
        self.optim = optim.Adam(self.network.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def L1_loss(self,inp,tar):
        return F.l1_loss(inp,tar)

    def cross_entropy_loss(self, inp, tar):
        loss = nn.CrossEntropyLoss()
        return loss(inp.reshape((inp.shape[0], params['num_class'], -1)), tar[:, 0, :, :].reshape(tar.shape[0], -1))

    def gen_loss(self,img,tar):
        return self.cross_entropy_loss(img, tar)
    # + self.L1_loss(img,tar.to(self.device,dtype=torch.float))

    def train_step(self,inp,tar):
        self.network.zero_grad()
        gen_opt = self.network(inp)

        g_loss = self.gen_loss(gen_opt, tar.to(self.device,dtype=torch.long))
        g_loss.backward()
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
                torch.save(self.network.state_dict(), save_path+str(epoch+1)+'.pth')
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss': G_loss,
                }, save_path+'_ckpt_'+str(epoch+1)+'.pth')
    

        print('training finished!')   
        print('epoch:%d,training steps:%d'%(self.num_epochs,iters))


        return self.network

start_time = time.time()
model = Model(batch_size, epoch, False, image_path, label_path, plane_view)
model.train()
print("--- %s seconds ---" % (time.time() - start_time))