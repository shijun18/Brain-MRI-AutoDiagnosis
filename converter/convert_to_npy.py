import os
import numpy as np
import shutil
from common_utils import save_as_hdf5, nii_reader



def convert_to_npy(input_path,save_path):
    '''
    Convert the raw data(e.g. dcm and *.nii) to numpy array and save as hdf5.
    '''
    
    for sub_path in os.scandir(input_path):
        if sub_path.is_dir():
            sub_save_path = os.path.join(save_path,sub_path.name)
            if os.path.exists(sub_save_path):
                shutil.rmtree(sub_save_path)
            os.makedirs(sub_save_path)
            convert_to_npy(sub_path.path,sub_save_path)

        else:
            if sub_path.is_file() and sub_path.name.endswith('.nii'):
                ID = sub_path.name.split('.')[0] 
                hdf5_path = os.path.join(save_path,ID + '.hdf5')
                _,image = nii_reader(sub_path.path)
                print(f'sample: {ID}, val max: {np.max(image)}, val min: {np.min(image)}, shape: {image.shape}')
                save_as_hdf5(image.astype(np.int16),hdf5_path,'image')



if __name__ == "__main__":

    # convert image to numpy array and save as hdf5
    input_path = '../dataset/nii_data/'
    save_path = '../dataset/npy_data/'
    convert_to_npy(input_path,save_path)

  