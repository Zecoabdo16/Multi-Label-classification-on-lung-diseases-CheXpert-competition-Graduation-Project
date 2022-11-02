import cv2
import numpy as np
import pandas as pd

import torch 
from torch.utils.data import Dataset
from numpy import prod, zeros, sqrt
from numpy.random import randn
from scipy.linalg import qr
from sklearn.metrics import mean_squared_error


def godec(X, rank=1, card=None, iterated_power=1, max_iter=100, tol=0.001):
    
    iter = 1
    RMSE = []
    card = prod(X.shape) if card is None else card
    
    X = X.T if(X.shape[0] < X.shape[1]) else X
    m, n = X.shape
    
    # Initialization of L and S
    L = X
    S = zeros(X.shape)
    LS = zeros(X.shape)
    
    while True:
        # Update of L
        Y2 = randn(n, rank)
        for i in range(iterated_power):
            Y1 = L.dot(Y2)
            Y2 = L.T.dot(Y1)
        Q, R = qr(Y2, mode='economic')
        L_new = (L.dot(Q)).dot(Q.T)
        
        # Update of S
        T = L - L_new + S
        L = L_new
        T_vec = T.reshape(-1)
        S_vec = S.reshape(-1)
        idx = abs(T_vec).argsort()[::-1]
        S_vec[idx[:card]] = T_vec[idx[:card]]
        S = S_vec.reshape(S.shape)
        
        # Reconstruction
        LS = L + S
        
        # Stopping criteria
        error = sqrt(mean_squared_error(X, LS))
        RMSE.append(error)
        
        #print("iter: ", iter, "error: ", error)
        if (error <= tol) or (iter >= max_iter):
            break
        else:
            iter = iter + 1

    return L, S, LS, RMSE


class MyGen(Dataset):
    def __init__(self,
                 csv_path='/content/df.csv',
                 image_size=320, 
                 use_clahe= False ,
                 use_fourier = False ,
                 use_histoeq = False,
                 use_decomposition = False,
                 use_godec = False,
                 use_top_bottom= True ):


        self.meta_data = pd.read_csv(csv_path)
        self.meta_data.reset_index(inplace=True, drop=True)       
        self._num_images = self.meta_data.shape[0]
        self.image_size = image_size
        self.use_clahe = use_clahe
        self.use_fourier = use_fourier
        self.use_histoeq = use_histoeq
        self.use_decomposition = use_decomposition
        self.use_top_bottom = use_top_bottom

    def fourier_filter(self, image):
        ham = np.hamming(256)[:, None] # 1D hamming
        ham2d = np.sqrt(np.dot(ham, ham.T))
        f = cv2.dft(image.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f)
        f_complex = f_shifted[:,:,0]*1j + f_shifted[:,:,1]
        f_filtered = ham2d * f_complex
        f_filtered_shifted = np.fft.fftshift(f_filtered)
        inv_img = np.fft.ifft2(f_filtered_shifted) # inverse F.T.
        filtered_img = np.abs(inv_img)
        filtered_img -= filtered_img.min()
        filtered_img = filtered_img*255 / filtered_img.max()
        filtered_img = filtered_img.astype(np.uint8)

        return filtered_img
    

    @property        
    def class_counts(self):
        return self.value_counts_dict
    
    @property
    def imbalance_ratio(self):
        return self.imratio


    @property
    def get_path(self):
        return np.array(self.meta_data.iloc[:,0]).reshape(-1, 1)
           
    @property  
    def data_size(self):
        return self._num_images    
    
    def __len__(self):
        return self._num_images
    
    def __getitem__(self, idx):
        image = cv2.imread(self.meta_data.iloc[idx, 0],0)
        
        image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=10)
            image = clahe.apply(image) 
            image = image.astype(np.float32)

        if self.use_fourier:
            image = self.fourier_filter(image)
            
        if self.use_histoeq:
            image = cv2.equalizeHist(image)

        if self.use_decomposition:
            decomposer = CartoonTextureDecomposition(sigma=2, ksize=5, n_iter=5, threshold_fun=None)
            _, image = decomposer.decompose(image)
        if self.use_top_bottom:
          kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
          # Top Hat Transform
          topHat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
          # Black Hat Transform
          blackHat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
          #Result
          image = image + topHat - blackHat




        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        image = image/255.0
        
        image = image.transpose((2, 0, 1)).astype(np.float32)
        
 
        return image
   