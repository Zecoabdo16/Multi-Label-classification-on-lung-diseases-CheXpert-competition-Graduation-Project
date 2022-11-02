import cv2
import torch 

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

class CheXpert(Dataset):
    '''
    Reference: 
        Large-scale Robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification
        Zhuoning Yuan, Yan Yan, Milan Sonka, Tianbao Yang
        International Conference on Computer Vision (ICCV 2021)
    '''
    def __init__(self, 
                 csv_path, 
                 image_root_path='',
                 image_size=320,
                 class_index=0, 
                 use_upsampling=True,
                 shuffle=True,
                 seed=123,
                 verbose=True,
                 upsampling_cols=['Cardiomegaly', 'Consolidation'],
                 train_cols=['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis',  'Pleural Effusion']):
        
    
        # load data from csv
        self.df = pd.read_csv(csv_path)
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0-small/', '')
        self.df['Path'] = self.df['Path'].str.replace('CheXpert-v1.0/', '')
            
        # upsample selected cols
        if use_upsampling:
            assert isinstance(upsampling_cols, list), 'Input should be list!'
            sampled_df_list = []
            for col in upsampling_cols:
                print ('Upsampling %s...'%col)
                sampled_df_list.append(self.df[self.df[col] == 1])
            self.df = pd.concat([self.df] + sampled_df_list, axis=0)

        
        self._num_images = len(self.df)  
            
        # shuffle data
        if shuffle:
            data_index = list(range(self._num_images))
            np.random.seed(seed)
            np.random.shuffle(data_index)
            self.df = self.df.iloc[data_index]
        
        
        #assert class_index in [-1, 0, 1, 2, 3, 4], 'Out of selection!'
        assert image_root_path != '', 'You need to pass the correct location for the dataset!'

        if class_index == -1: # 5 classes
            if verbose:
                print ('Multi-label mode: True, Number of classes: [%d]'%len(train_cols))
                print ('-'*30)
            self.select_cols = train_cols
            self.value_counts_dict = {}
            for class_key, select_col in enumerate(train_cols):
                class_value_counts_dict = self.df[select_col].value_counts().to_dict()
                self.value_counts_dict[class_key] = class_value_counts_dict
        else:       # 1 class
            self.select_cols = [train_cols[class_index]]  # this var determines the number of classes
            self.value_counts_dict = self.df[self.select_cols[0]].value_counts().to_dict()
        
        self.mode = mode
        self.class_index = class_index
        self.image_size = image_size
        self.transforms = transforms
        
        self._images_list =  [image_root_path+path for path in self.df['Path'].tolist()]
        if class_index != -1:
            self._labels_list = self.df[train_cols].values[:, class_index].tolist()
        else:
            self._labels_list = self.df[train_cols].values.tolist()
    
        if True:
            if class_index != -1:
                if flip_label:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[-1]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[-1] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
                else:
                    self.imratio = self.value_counts_dict[1]/(self.value_counts_dict[0]+self.value_counts_dict[1])
                    if verbose:
                        print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[1], self.value_counts_dict[0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(self.select_cols[0], class_index, self.imratio ))
                        print ('-'*30)
            else:
                imratio_list = []
                for class_key, select_col in enumerate(train_cols):
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
                    if verbose:
                        #print ('-'*30)
                        print('Found %s images in total, %s positive images, %s negative images'%(self._num_images, self.value_counts_dict[class_key][1], self.value_counts_dict[class_key][0] ))
                        print ('%s(C%s): imbalance ratio is %.4f'%(select_col, class_key, imratio ))
                        print ()
                        #print ('-'*30)
                self.imratio = np.mean(imratio_list)
                self.imratio_list = imratio_list
                
            
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
    
    def histogram_equalization(self, image):
        r,g,b = cv2.split(image)
        r = cv2.equalizeHist(r)
        g = cv2.equalizeHist(g)
        b = cv2.equalizeHist(b)
        img = np.stack((r,g,b), -1)
        return img
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):

        image = cv2.imread(self._images_list[idx], 0)
        image = cv2.cvtColor(image, cv2.COLOR_GBR2RGB)
        
        # resize and normalize; e.g., ToTensor()
        image = cv2.resize(image, dsize=(self.image_size, self.image_size)) 
        image = histogram_equalization(image)
        image = image/255.0
        image = image.transpose((2, 0, 1)).astype(np.float32)
        label = np.array(self._labels_list[idx]).reshape(-1).astype(np.float32)
        return image, label