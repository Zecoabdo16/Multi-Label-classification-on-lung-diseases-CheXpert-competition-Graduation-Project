import cv2
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset

class MyGen(Dataset):
    '''
    Reference: 
        Large-scale Robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification
        Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang
        International Conference on Computer Vision (ICCV 2021)
    '''
    def __init__(self,
                 image_root_path, 
                 csv_path, 
                 image_size, 
                 frontal,       
                 col_to_train=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion']):
        
    
        self.meta_data = pd.read_csv(image_root_path+csv_path)
        
        if frontal:
            self.meta_data = self.meta_data[self.meta_data['Frontal/Lateral']=='Frontal'].reset_index(drop=True)
        
        self.select_cols = col_to_train
        self.image_size = image_size

    def num_classes(self):
        return len(self.select_cols)
       
    def data_size(self):
        return self.meta_data.shape[0]    
    
    def __len__(self):
        return self.meta_data.shape[0]
 
    
    def __getitem__(self, idx):
        image = cv2.imread(self.meta_data.iloc[idx, 0], 0)
            
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)             
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)/255.0
        image = image.transpose((2, 0, 1)).astype(np.float32)
        
        label = self.meta_data.loc[idx, self.select_cols].values.astype(np.float32)
        
        return image, label
   