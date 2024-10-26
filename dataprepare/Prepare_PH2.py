
"""
PH2 official data provided by the '.bmp' format, should first batch modify the suffix named '.jpg' or '.png' format before processing.
"""

import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob

# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare PH2 dataset #################################################
Dataset_add = './PH2/'
Tr_add = 'images'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
Data_train_PH2    = np.zeros([200, height, width, channels])
Label_train_PH2   = np.zeros([200, height, width])

print('Reading PH2')
print(Tr_list)
for idx in range(len(Tr_list)):
    print(idx+1)
    img = sc.imread(Tr_list[idx])
    img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
    Data_train_PH2[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[len(b)-16: len(b)-4] 
    add = (a+ 'masks/' + b +'.png')    
    img2 = sc.imread(add)
    img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train_PH2[idx, :,:] = img2
         
print('Reading PH2 finished')

Train_img      = Data_train_PH2[0:140,:,:,:]
Validation_img = Data_train_PH2[140:140+20,:,:,:]
Test_img       = Data_train_PH2[140+20:200,:,:,:]

Train_mask      = Label_train_PH2[0:140,:,:]
Validation_mask = Label_train_PH2[140:140+20,:,:]
Test_mask       = Label_train_PH2[140+20:200,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)


