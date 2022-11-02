import cv2
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset

class MyGen_1(Dataset):

    def __init__(self,
                 csv_path,
                 image_size
                 ):
        
    
        self.meta_data = pd.read_csv(csv_path)

        
        self.image_size = image_size

    def num_classes(self):
        return len(self.select_cols)
       
    def data_size(self):
        return self.meta_data.shape[0]    
    
    def __len__(self):
        return self.meta_data.shape[0]


    @property
    def get_path(self):
        return np.array(self.meta_data.iloc[:,0]).reshape(-1, 1)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.meta_data.iloc[idx, 0], 0)
            
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)             
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)/255.0
        image = image.transpose((2, 0, 1)).astype(np.float32)
        

        return image
   