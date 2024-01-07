"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet
from torch.utils.tensorboard import SummaryWriter

from utils.utils import med_reshape
#from utils.utils import adapt_input_tensor_to_unet


import torch.nn.functional as F

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device
        self.tensorboard_test_writer = SummaryWriter(comment="_test")
        #self.model_path = ''
        #Uncaomment for test
        self.model_path = '/home/workspace/out/model/model.pth'
        #self.images_path = '/home/workspace/home/tests/images/'
        self.batch_size = 64
        
        if model is None:
            self.model = UNet(num_classes=3)

        if self.model_path:
            print('load model...')
            self.model.load_state_dict(torch.load(self.model_path , map_location=self.device))
            print('model loaded')
            # print(self.model)
        self.model.to(device)

        
    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()

        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # <YOUR CODE HERE>
        for i in range(volume.shape[0]):
            slice = volume[i,:,:]
            width = volume.shape[1]
            height = volume.shape[2]
            slice= slice.reshape(1,width, height)
            #cast from torch.float64 to torch.float32 to ensure compability with parameter of the model and
            #avoid RuntimeError: Expected object of scalar type Double but got scalar type Float for argument #3 'mat1'                 #in call to _th_addmm_
            image_tensor= torch.unsqueeze(torch.tensor(slice),dim=0)  
            data = image_tensor.to(torch.float32)
            print(data.size())
            #data = adapt_input_tensor_to_unet(data,self.model)
            if torch.cuda.is_available():
                data = data.to(self.device)
            prediction = self.model(data)
            #print(prediction.size())
            prediction_softmax = F.softmax(prediction, dim=1) 
            prediction_softmax_np = prediction_softmax.to('cpu').detach().numpy()
            prediction_np = (prediction_softmax_np[0,0,:,:]<0.95).reshape(1,prediction_softmax_np.shape[2], prediction_softmax_np.shape[3]).astype(int) 
            slices.append(prediction_np)
        # Convert the list of slices to a 3D NumPy array
        prediction_mask = np.concatenate(slices, axis=0)
        return prediction_mask 
    
    
    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        width = 64
        height = volume.shape[2]
        if width % height != 0:
           height = width
        volume_prime = med_reshape(volume, (64,width,height))
        #volume_prime = med_reshape(volume, (64,64,64))
        return(self.single_volume_inference(volume_prime))