"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        
        raise NotImplementedError

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
            #cast from torch.float64 to torch.float32 to ensure compability with parameter of the model and
            #avoid RuntimeError: Expected object of scalar type Double but got scalar type Float for argument #3 'mat1'                 #in call to _th_addmm_
            image_tensor= torch.unsqueeze(torch.tensor(slice),dim=0)                      
            data = torch.unsqueeze(image_tensor,dim=0).to(torch.float32) 
            if torch.cuda.is_available():
                data = data.to(self.device)
            prediction = self.model(data)
            prediction = prediction.squeeze().to('cpu').detach().numpy().astype(int)
            threshold = 0.5
            prediction = (prediction > threshold).astype(int)
            slices.append(prediction[0,:,:].reshape(1,prediction.shape[1],prediction.shape[2]))
        # Convert the list of slices to a 3D NumPy array
        prediction_mask = np.concatenate(slices, axis=0)
        return prediction_mask 
