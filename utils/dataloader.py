
"""
RGB color code and object categories:
------------------------------------
000 BW: Background waterbody
001 HD: Human divers
010 PF: Plants/sea-grass
011 WR: Wrecks/ruins
100 RO: Robots/instruments
101 RI: Reefs and invertebrates
110 FV: Fish and vertebrates
111 SR: Sand/sea-floor (& rocks)
"""

from __future__ import print_function,division
import os
from os.path import join, exists
from torch.utils.data import DataLoader,Dataset
import torch
import torchvision
from torchvision import datasets,models,transforms
import itertools 

def robotfishhumanreefwrecks(mask):
    human=torch.zeros((imw,imh))
    fish=torch.zeros((imw,imh))
    robot=torch.zeros((imw,imh))
    reef=torch.zeros((imw,imh))
    wrecks=torch.zeros((imw,imh))
    for i in range(imw):
        for j in range(imh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                    Human[i, j] = 1 
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                fish[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                robot[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==1):
                reef[i, j] = 1  
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                wreck[i, j] = 1 
            else:
                pass

    return torch.stack((robot,fish,human,reef,wreck),-1)

def getSaliency(mask):
    imgw,imgh=mask.shape[0],mask.shape[1]
    sal=torch.zeros((imgw,imgh))
    for i in range(imgw):
        for j in range(imgh):
            if (mask[i,j,0]==0 and mask[i,j,1]==0 and mask[i,j,2]==1):
                sal[i,j]=1
            elif (mask[i,j,0]==1 and mask[i,j,1]==0 and mask[i,j,2]==0):
                sal[i, j] = 1  
            elif (mask[i,j,0]==1 and mask[i,j,1]==1 and mask[i,j,2]==0):
                sal[i, j] = 1   
            elif (mask[i,j,0]==0 and mask[i,j,1]==1 and mask[i,j,2]==1):
                sal[i, j] = 0.8  
            else: pass  
        
            
    return sal.unsqueeze(1)
            
            

def processData(image,mask,sal=False):
     # scaling image data and masks
    image = image / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    m = []
    for i in range(mask.shape[0]):
        if sal:
            m.append(getSaliency(mask[i]))
        else:
            m.append(robotfishhumanreefwrecks(mask[i]))
            
    m=torch.tensor(m)
    return (img,m)

# def load_training(batch_size,train_path,image_folder,mask_folder,aug_dict, target_size=(256,256), sal=False):
    
#     transform = transforms.Compose(
#         [transforms.RandomHorizontalFlip(),
#          transforms.ToTensor()])
#     data=datasets.ImageFolder(**aug_dict,transform=transform)
    
#     image_generator=torch.utils.data.DataLoader(data,train_path,prefix='image',batch_size=batch_size,target_size=target_size)
#     mask_generator=torch.utils.data.DataLoader(data,train_path,prefix='mask',batch_size=batch_size,target_size=target_size)
    
#     for (img, mask) in it.izip(image_generator, mask_generator):
#         img, mask_indiv = processSUIMDataRFHW(img, mask, sal)
#         yield (img, mask_indiv)

class SUIMData(Dataset):
    def __init__(self,images_filenames,images_directory,marks_directory,transform=None):
        self.images_filenames=images_filenames
        self.images_directory=images_directory
        self.masks_directory=masks_directory
        self.transform=transform
    def __len__(self):
        return len(self.images_filenames)
    
    def __getitem(self,idx):
        image_filename=self.images_filenames[idx]
        image=cv2.imread(os.path.join(self.images_directory,image_filename))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        mask=cv2.imread(os.path.join(self.masks_directory,image_filename))
        
        image=processData(image)
        mask=processData(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

def load_training(batch_size,image_folder,mask_folder, sal=False):
    scale=Rescale(256)
    transform = transforms.Compose(
#         [transforms.RandomHorizontalFlip(),
        [Rescale(256),
         transforms.ToTensor()])
    image_data=torchvision.datasets.ImageFolder(root=image_folder,transform=transform)
    mask_data=torchvision.datasets.ImageFolder(root=mask_folder,transform=transform)

    image_generator=DataLoader(image_data,batch_size=batch_size,shuffle=True,num_workers=4)
    mask_generator=DataLoader(mask_data,batch_size=batch_size,shuffle=True,num_workers=4)
    for (img,mask) in zip(image_generator,mask_generator):
        img,mask_indiv=processData(img,mask,sal)
        print(img.shape,mask_indiv.shape)
        
def getPaths(data_dir):
    # read image files from directory
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG', '*.bmp']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return image_paths
