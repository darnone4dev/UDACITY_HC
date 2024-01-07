"""
Various utility methods in this module
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from PIL import Image

# Tell Matplotlib to not try and use interactive backend
mpl.use("agg")

def mpl_image_grid(images):
    """
    Create an image grid from an array of images. Show up to 16 images in one figure

    Arguments:
        image {Torch tensor} -- NxWxH array of images

    Returns:
        Matplotlib figure
    """
    # Create a figure to contain the plot.
    n = min(images.shape[0], 16) # no more than 16 thumbnails
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2*rows, 2*cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        # Start next subplot.
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            # this is specifically for 3 softmax'd classes with 0 being bg
            # We are building a probability map from our three classes using 
            # fractional probabilities contained in the mask
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y])*vol[1,x,y], (1-vol[0,x,y])*vol[2,x,y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: # plotting only 1st channel
            plt.imshow((images[i, 0]*255).int(), cmap= "gray")

    return figure

def log_to_tensorboard(writer, loss, data, target, prediction_softmax, prediction, counter):
    """Logs data to Tensorboard

    Arguments:
        writer {SummaryWriter} -- PyTorch Tensorboard wrapper to use for logging
        loss {float} -- loss
        data {tensor} -- image data
        target {tensor} -- ground truth label
        prediction_softmax {tensor} -- softmax'd prediction
        prediction {tensor} -- raw prediction (to be used in argmax)
        counter {int} -- batch and epoch counter
    """
    writer.add_scalar("Loss",\
                    loss, counter)
    writer.add_figure("Image Data",\
        mpl_image_grid(data.float().cpu()), global_step=counter)
    writer.add_figure("Mask",\
        mpl_image_grid(target.float().cpu()), global_step=counter)
    writer.add_figure("Probability map",\
        mpl_image_grid(prediction_softmax.cpu()), global_step=counter)
    writer.add_figure("Prediction",\
        mpl_image_grid(torch.argmax(prediction.cpu(), dim=1, keepdim=True)), global_step=counter)

def save_numpy_as_image(arr, path):
    """
    This saves image (2D array) as a file using matplotlib

    Arguments:
        arr {array} -- 2D array of pixels
        path {string} -- path to file
    """
    plt.imshow(arr, cmap="gray") #Needs to be in row,col order
    plt.savefig(path)

def med_reshape(image, new_shape):
    """
    This function reshapes 3D data to new dimension padding with zeros
    and leaving the content in the top-left corner

    Arguments:
        image {array} -- 3D array of pixel data
        new_shape {3-tuple} -- expected output shape

    Returns:
        3D array of desired shape, padded with zeroes
    """

    reshaped_image = np.ones(new_shape)

    # TASK: write your original image into the reshaped image
    # <CODE GOES HERE>
    if image is  None: 
        raise ValueError('Image parameter empty')
    if new_shape is  None:
        raise ValueError('New_shape parameter empty') 
    ## Check 3D image shape 
    if  len(image.shape) != 3:
        raise ValueError("Image isn't a 3D image")
    ## Check source and target shape
    if len(new_shape) != len(image.shape):
        raise ValueError(f"Target shape {len(new_shape)} isn't equal to image shape {len(image.shape)}") 

    X_s = image.shape[0]
    Y_s = image.shape[1]  
    Z_s = image.shape[2]

    X_t = new_shape[0]
    Y_t = new_shape[1]
    Z_t = new_shape[2]

    ## Check that source image can be embedded within target image 
    if X_s > X_t:
       raise ValueError(f"Image source bigger than target X: {X_s} > {X_t}")
    if Y_s > Y_t:
       raise ValueError(f"Image source bigger than target Y: {Y_s} > {Y_t}")
    if Z_s > Z_t:
       raise ValueError(f"Image source bigger than target Z: {Z_s} > {Z_t}")

    for i in range(X_s):
       for j in range(Y_s):
          for k in range(Z_s):
            reshaped_image[i,j,k] = image[i,j,k]   
            
    return reshaped_image


def adapt_input_tensor(input_tensor, model):
    # Get the expected input size from the model's first layer
    expected_input_size = list(model.conv1.weight.size())
    expected_input_size[1] = input_tensor.size(1)  # Update the channel dimension

    # Resize the input tensor to match the expected size
    adapted_input_tensor = torch.nn.functional.interpolate(input_tensor, size=expected_input_size[2:], mode='nearest')

    return adapted_input_tensor



