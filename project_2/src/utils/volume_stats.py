"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expection: 3 dimensional inputs, got {len(a.shape)} and {len(b.shape)}")

    if len(a.shape) != len(b.shape):
        raise Exception(f"Expection: inputs of the same shape, got {len(a.shape)} and {len(b.shape)}")
    
    if a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1] or b.shape[2] != b.shape[2]:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")
        
    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    # <YOUR CODE HERE>
    intersection = np.sum(a*b)
    volumes      =   np.sum(a)  + np.sum(b)
    if volumes == 0: return -1
    coeff = 2*float(intersection)/float(volumes) 
    return(coeff)


    
    
    

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {len(a.shape)} and {len(b.shape)}")

    if len(a.shape) != len(b.shape):
        raise Exception(f"Expecting inputs of the same shape, got {len(a.shape)} and {len(b.shape)}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # <YOUR CODE GOES HERE>
    intersection = np.sum(a == b)
    union      =  np.sum(a != b )*2 + intersection
    if union == 0: return -1
    coeff = intersection/ union 
    return(coeff)