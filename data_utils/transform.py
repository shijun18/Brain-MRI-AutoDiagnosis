import torch
import numpy as np
from skimage.transform import resize, warp
from transforms3d.euler import euler2mat
from transforms3d.affines import compose


class Trunc_and_Normalize(object):
    '''
    truncate gray scale and normalize to [0,1]
    '''
    def __init__(self, scale=None):
        self.scale = scale
        if self.scale is not None:
            assert len(self.scale) == 2, 'scale error'

    def __call__(self, sample):
        image = sample['image']

        if self.scale is not None:
            # gray truncation
            image = image - self.scale[0]
            gray_range = self.scale[1] - self.scale[0]
            image[image < 0] = 0
            image[image > gray_range] = gray_range
        else:
            gray_range = np.max(image)
        # normalize
        image = image / gray_range

        sample['image'] = image
        return sample


class CropResize(object):
    '''
    Data preprocessing.
    Adjust the size of input data to fixed size by cropping and resize
    Args:
    - dim: tuple of integer, fixed size
    - crop: single integer, factor of cropping, H/W ->[:,crop:-crop,crop:-crop]
    '''
    def __init__(self, dim=None, crop=0):
        self.dim = dim
        self.crop = crop

    def __call__(self, sample):

        # image: 3d numpy array, (D,H,W)
        # label: integer, 0,1,...
        image = sample['image']
        # resize
        if self.dim is not None and image.shape != self.dim:
            # crop
            if self.crop is not None and self.crop != 0:
                image = image[:, self.crop:-self.crop, self.crop:-self.crop]
            image = resize(image, self.dim, anti_aliasing=True)

        sample['image'] = image

        return sample


class RandomTranslationRotationZoom(object):
    '''
    Data augmentation method.
    Including random translation, rotation and zoom, which keep the shape of input.
    Args:
    - mode: string, consisting of 't','r' or 'z'. Optional methods and 'trz'is default.
            't'-> translation,
            'r'-> rotation,
            'z'-> zoom.
    '''
    def __init__(self, mode='trz'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array, (D,H,W)

        image = sample['image']
        # get transform coordinate
        img_size = image.shape
        coords0, coords1, coords2 = np.mgrid[:img_size[0], :img_size[1], :
                                             img_size[2]]
        coords = np.array([
            coords0 - img_size[0] / 2, coords1 - img_size[1] / 2,
            coords2 - img_size[2] / 2
        ])
        tform_coords = np.append(coords.reshape(3, -1),
                                 np.ones((1, np.prod(img_size))),
                                 axis=0)
        # transform configuration
        # translation
        if 't' in self.mode:
            translation = [
                0, np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ]
        else:
            translation = [0, 0, 0]

        # rotation
        if 'r' in self.mode:
            rotation = euler2mat(
                np.random.uniform(-5, 5) / 180.0 * np.pi, 0, 0, 'sxyz')
        else:
            rotation = euler2mat(0, 0, 0, 'sxyz')

        # zoom
        if 'z' in self.mode:
            zoom = [
                1, np.random.uniform(0.9, 1.1),
                np.random.uniform(0.9, 1.1)
            ]
        else:
            zoom = [1, 1, 1]

        # compose
        warp_mat = compose(translation, rotation, zoom)

        # transform
        w = np.dot(warp_mat, tform_coords)
        w[0] = w[0] + img_size[0] / 2
        w[1] = w[1] + img_size[1] / 2
        w[2] = w[2] + img_size[2] / 2
        warp_coords = w[0:3].reshape(3, img_size[0], img_size[1], img_size[2])

        image = warp(image, warp_coords)

        sample['image'] = image

        return sample


class RandomFlip(object):
    '''
    Data augmentation method.
    Flipping the image, including horizontal and vertical flipping.
    Args:
    - mode: string, consisting of 'h' and 'v'. Optional methods and 'hv' is default.
            'h'-> horizontal flipping,
            'v'-> vertical flipping,
            'hv'-> random flipping.

    '''
    def __init__(self, mode='hv'):
        self.mode = mode

    def __call__(self, sample):
        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        image = sample['image']

        if 'h' in self.mode and 'v' in self.mode:
            if np.random.uniform(0, 1) > 0.5:
                image = image[:, ::-1, ...]
            else:
                image = image[..., ::-1]

        elif 'h' in self.mode:
            image = image[:, ::-1, ...]

        elif 'v' in self.mode:
            image = image[..., ::-1]
        # avoid the discontinuity of array memory
        image = image.copy()
        sample['image'] = image
        return sample


class GaussNoise(object):
    '''
    Data augmentation method.
    Adding gaussian noise to image.
    Args:
    - mean: float
    - var: float
    '''
    def __init__(self, mean=0.0, var=0.01):
        self.mean = mean
        self.std = var**0.5

    def __call__(self, sample):
        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        image = sample['image']

        gauss_noise = np.random.normal(self.mean, self.std, image.shape)
        image = image + gauss_noise
        image = np.clip(image, 0.0, 1.0)

        sample['image'] = image
        return sample


class To_Tensor(object):
    '''
    Convert the data in sample to torch Tensor.
    Args:
    - n_class: the number of class
    '''
    def __call__(self, sample):

        # image: numpy array, (D,H,W)
        # label: integer, 0,1,..
        image = sample['image']
        # expand dim for image -> (C,D,H,W)
        image = np.expand_dims(image, axis=0)
        # convert to Tensor
        sample['image'] = image
        return sample
