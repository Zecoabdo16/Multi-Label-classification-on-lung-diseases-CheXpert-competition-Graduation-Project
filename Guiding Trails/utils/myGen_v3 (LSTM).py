import cv2
import numpy as np
import pandas as pd
from scipy import linalg

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
                 use_clahe,
                 n_patch,
                 col_to_train):
        
    
        self.meta_data = pd.read_csv(image_root_path+csv_path)
         
        if frontal:
            self.meta_data = self.meta_data[self.meta_data['Frontal/Lateral']=='Frontal'].copy()
        
        self.meta_data.reset_index(inplace=True, drop=True)
        
        self.select_cols = col_to_train
        self.value_counts_dict = {}
        for class_key, select_col in enumerate(self.select_cols):
            class_value_counts_dict = self.meta_data[select_col].value_counts().to_dict()
            self.value_counts_dict[class_key] = class_value_counts_dict

        imratio_list = []
        for class_key, select_col in enumerate(self.select_cols):
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
        self.use_clahe = use_clahe
        self.n_patch = n_patch

    def get_patches(self, image):
        new_dim = self.image_size//(self.n_patch//2)

        X = np.zeros((self.n_patch, new_dim*new_dim))
        for r in range(self.n_patch//2):
            for c in range(self.n_patch//2):
                idx = r*2+c
                start_edge_r = r*new_dim
                end_edge_r = (r+1)*new_dim
                start_edge_c = c*new_dim
                end_edge_c = (c+1)*new_dim

                X[idx, :] = image[start_edge_r:end_edge_r, start_edge_c:end_edge_c].ravel()

        return X

    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio

    @property
    def num_classes(self):
        return len(self.select_cols)
       
    @property  
    def data_size(self):
        return self._num_images    
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):
        image = cv2.imread(self.meta_data.iloc[idx, 0], 0)
        
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=10)
            image = clahe.apply(image) #*0.75
            image = image.astype(np.float32)
            
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        image = image/255.0
        
        patches = self.get_patches(image).astype(np.float32)
        
        label = self.meta_data.loc[idx, self.select_cols].values.astype(np.float32)
        return patches, label
   