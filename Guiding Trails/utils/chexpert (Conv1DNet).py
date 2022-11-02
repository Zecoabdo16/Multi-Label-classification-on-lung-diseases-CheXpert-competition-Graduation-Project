import cv2
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset

class CheXpert(Dataset):
    '''
    Reference: 
        Large-scale Robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification
        Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang
        International Conference on Computer Vision (ICCV 2021)
    '''
    def __init__(self,
                 image_root_path, 
                 csv_path, 
                 image_size=320):
        
    
        self.meta_data = pd.read_csv(image_root_path+csv_path)

        self.select_cols = self.meta_data.columns[1:]
        self.value_counts_dict = {}
        for class_key, select_col in enumerate(self.select_cols):
            class_value_counts_dict = self.meta_data[select_col].value_counts().to_dict()
            self.value_counts_dict[class_key] = class_value_counts_dict

        imratio_list = []
        for class_key, select_col in enumerate(self.meta_data.columns[1:]):
            try:
                imratio = self.value_counts_dict[class_key][1]/(self.value_counts_dict[class_key][0]+self.value_counts_dict[class_key][1])
            except:
                if len(self.value_counts_dict[class_key]) == 1 :
                    only_key = list(self.value_counts_dict[class_key].keys())[0]
                    if only_key == 0:
                        self.value_counts_dict[class_key][1] = 0
                        imratio = 0 # no postive samples
                    else:    
                        self.value_counts_dict[class_key][1] = 0
                        imratio = 1 # no negative samples
                    
            imratio_list.append(imratio)

        self.imratio = np.mean(imratio_list)
        self.imratio_list = imratio_list 

        self._num_images = self.meta_data.shape[0]

        self.image_size = image_size

    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.meta_data.columns[1:])
       
    @property  
    def data_size(self):
        return self._num_images    
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):
        image = cv2.imread(self.meta_data.iloc[idx, 0], 0)
        
        clahe = cv2.createCLAHE(clipLimit=10)
        image = clahe.apply(image) #*0.75
        
        image = image.astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        image = image/255.0
        
        image = image.transpose((2, 0, 1)).astype(np.float32)
        
        label = self.meta_data.iloc[idx, 1:].values.astype(np.float32)
        return image, label

