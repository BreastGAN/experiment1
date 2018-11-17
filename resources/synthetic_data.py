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
from dotmap import DotMap
import skimage.filters


# Do not modify img!
def gen_element(img, center, min_rad, max_rad, max_thresh, eps=0.1):

    def is_inside(x, y, center, rad1, rad2):
        x0 = (x - center[0])
        y0 = (y - center[1])
        return x0**2 / rad1 + y0**2 / rad2 <= 1 + np.random.normal(0, eps)

    mask = np.zeros_like(img)
    min_thresh = 0.9  # smallest number I can actually see in the resulting image

    max_area = max_rad**2
    # print("max_area =", max_area)
    min_area = max(1, min_rad**2)
    # print("min_area =", min_area)
    area = np.random.uniform(min_area, max_area)
    # print("area =", area)

    rad1 = np.random.randint(min_rad, max_rad + 1)
    # print("r1 =", rad1)
    rad2 = max(1, int(area / rad1))
    area = rad1 * rad2
    # print("area =", area)
    # print("r2 =", rad2)

    for x in range(max(0, center[0] - rad1), min(center[0] + rad1, img.shape[0])):
        for y in range(max(0, center[1] - rad2), min(center[1] + rad2, img.shape[1])):
            if is_inside(x, y, center, rad1, rad2):
                mask[x, y] = 1

    # print("max_rad =", max_rad)
    thresh = area  # x in [0, max_area]
    thresh = thresh / max_area  # x in [0, 1]
    # print("ratio =", thresh)
    thresh = 1 - thresh
    thresh = (thresh * (max_thresh - min_thresh)) + min_thresh  # x in [min_thresh, max_thresh]
    # print("thresh =", thresh)
    mask *= thresh

    return mask


# doesn't work well under (30, 30)
def generate_synth(size=(256, 256), max_thresh=2.5):  # $max_thresh sigma

    def gen_img(size, max_thresh, masses=1):
        img_meta = DotMap()
        img_meta.laterality = 'L'
        img_meta.view = 'CC'

        img = np.random.standard_normal(size=size)
        img = skimage.filters.gaussian(img, sigma=1)

        min_rad = max(1, int(min(size) / 80))
        max_rad = max(min_rad, int(min(size) / 10))

        mask = np.zeros(size)
        for i in range(masses):
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])
            mask += gen_element(img, (x, y), min_rad, max_rad, max_thresh)

        return img, mask, img_meta

    while True:
        yield gen_img(size=size, max_thresh=max_thresh, masses=1)


def read_synth(n_samples, size=(256, 256), no_gpu=False):
    # In this case, synth data + batch and shuffle it. In our case, it will be quite different.
    imgs = np.zeros((n_samples, size[0], size[1]))
    masks = np.zeros((n_samples, size[0], size[1]))
    data_gen = generate_synth(size=size)

    for i in range(n_samples):
        imgs[i], masks[i], _ = next(data_gen)

    # imgs = np.reshape(imgs, (n_samples, size[0] * size[1]))
    # masks = np.reshape(masks, (n_samples, size[0] * size[1]))
    return imgs, masks
