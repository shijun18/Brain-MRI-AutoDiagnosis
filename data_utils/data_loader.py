import sys
sys.path.append('..')
from converter.common_utils import hdf5_reader
from torch.utils.data import Dataset


class DataGenerator(Dataset):
    '''
    Custom Dataset class for data loader.
    Args：
    - path_list: list of file path
    - label_dict: dict, file path as key, label as value
    - transform: the data augmentation methods
    '''
    def __init__(self, path_list, label_dict=None, transform=None, use_roi=False, repeat_factor=1.0):

        self.path_list = path_list
        self.label_dict = label_dict
        self.transform = transform
        self.use_roi = use_roi
        self.repeat_factor = repeat_factor

    def __len__(self):
        return int(len(self.path_list)*self.repeat_factor)

    def __getitem__(self, index):
        # Get image and label
        # image: D,H,W
        # label: integer, 0,1,..
        index = index % len(self.path_list)
        # print(f"index :{index}")
        image = hdf5_reader(self.path_list[index], 'image')
        if self.use_roi:
            mask = hdf5_reader(self.path_list[index], 'mask')
            image = self.get_roi(image,mask)
        
        sample = {'image': image}
        if self.transform is not None:
            sample = self.transform(sample)

        if self.label_dict is not None:
            label = self.label_dict[self.path_list[index]]
            sample['label'] = int(label)
        
        return sample

    #TODO
    def get_roi(self,img,mask):
        pass