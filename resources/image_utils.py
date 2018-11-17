# Copyright 2018 Lukas Jendele and Ondrej Skopek. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from PIL import Image
import pydicom
import skimage.transform


def to_numpy(img):
    width, height = img.size
    return np.reshape(img.getdata(), (height, width)).astype(np.float64)


def load_dicom(path):
    dcm = pydicom.dcmread(path)
    img = dcm.pixel_array
    return np.asarray(img).astype(np.float64)


def load_tif(path):
    return to_numpy(Image.open(path))


def load_image(path):
    if path.endswith('.dcm'):
        return load_dicom(path)
    elif path.endswith('.tif'):
        return load_tif(path)
    else:
        raise ValueError('Unknown file format for file {}'.format(path))


def standardize(img, img_meta):
    if img_meta.laterality == 'R':  # horizontal flip for right breasts
        img = np.flip(img, axis=1)  # TODO(#6): Might not be what we want
    # TODO(#6) what about MLO view?
    return img


def normalize_gaussian(img, mean=None, std=None):
    """
    Normalizes an image by the mean and dividing by std (Gaussian normalization).

    If mean or std is None, uses np.mean or np.std respectively.
    """
    if mean is None:
        mean = np.mean(img)
    if std is None:
        std = np.std(img)
    img = img - mean
    img = img / std  # Needed for comparable histograms!
    return img


def normalize(img, new_min=0, new_max=255):
    """
    Normalizes an image by linear transformation into the interval [new_min, new_max].
    """
    old_min = np.min(img)
    old_max = np.max(img)
    img = (img - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    return img


def downsample(img, size=256):
    scale = size / max(img.shape)
    img = skimage.transform.rescale(img, scale, mode='constant')
    img_new = np.zeros((size, size))
    img_new[:img.shape[0], :img.shape[1]] = img  # Fill in to full size x size # TODO(#6) Is this a good idea?
    return img_new
