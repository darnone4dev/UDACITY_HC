"""
Module loads the hippocampus dataset into RAM
"""
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from medpy.io import load

from utils.utils import med_reshape

def min_max_scaling(image):
    min_val = np.min(image)
    max_val = np.max(image)    
    norm_image = (image-min_val)/(max_val - min_val)
    return(norm_image)

def LoadHippocampusData(root_dir, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')

    images = [f for f in listdir(image_dir) if (
        isfile(join(image_dir, f)) and f[0] != ".")]

    out = []
    

    for f in images:

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))

        # TASK: normalize all images (but not labels) so that values are in [0..1] range
        # <YOUR CODE GOES HERE>
        image = min_max_scaling(image)
        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here

        # TASK: med_reshape function is not complete. Go and fix it!
        #print(f'file {f} - image shape {image.shape} - label shape {label.shape}')
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        #image = med_reshape(image, new_shape=(x_shape, y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)
        #label = med_reshape(label, new_shape=(x_shape, y_shape, z_shape)).astype(int)
        # TASK: Why do we need to cast label to int?
        # ANSWER: Appropriate with U-net ( used with PyTorch) used for segmentation tasks, make it compatible with the loss function
        #         and increase efficiency and memory management 

        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")
    return np.array(out)
