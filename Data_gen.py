import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A

NUM_WORKERS=0
PIN_MEMORY=True


transform = A.Compose([
    A.RandomCrop(width=640, height=640)
])

class Dataset_(Dataset):
    def __init__(self, image_dir,mask_dir,sc_dir,transform=transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sc_dir=sc_dir
        self.images = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index][:-4]+'_gt.npy')
        sc_path = os.path.join(self.sc_dir, self.images[index][:-4]+'sc_gt.npy')
        
        image = np.load(img_path,allow_pickle=True, fix_imports=True)
        mean=np.mean(image)
        std=np.std(image)
        image=(image-mean)/std
        
        mask = np.load(mask_path,allow_pickle=True, fix_imports=True)
        mask[np.where(mask>0)]=1.0
        
        
        scars = np.load(sc_path,allow_pickle=True, fix_imports=True)
        scars[np.where(scars>0)]=1.0
                
        if image.shape[0]==576:
         temp=np.zeros([640,640])
         temp1=np.zeros([640,640])
         temp2=np.zeros([640,640])
         
         temp[32:608, 32:608] = image
         image=temp
         
         temp1[32:608, 32:608] = mask
         mask=temp1
         
         temp2[32:608, 32:608] = scars
         scars=temp2
         
  
        image=np.expand_dims(image, axis=0)
        mask=np.expand_dims(mask, axis=0)
        scars=np.expand_dims(scars, axis=0)

        return image,mask,scars,self.images[index][:-4]
    
def Data_Loader( test_dir,test_maskdir,sc_dir,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    
    test_ids = Dataset_( image_dir=test_dir, mask_dir=test_maskdir,sc_dir=sc_dir)

    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    
    return data_loader

