"""
This file contains code that will kick off training and testing processes
"""
import os
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils.utils import log_to_tensorboard

#from expfrom torch.utils.tensorboard import SummaryWriter

from data_prep.HippocampusDatasetLoader import LoadHippocampusData
from networks.RecursiveUNet import UNet
from inference.UNetInferenceAgent import UNetInferenceAgent
from utils.volume_stats import Dice3d, Jaccard3d
from data_prep.HippocampusDatasetLoader import min_max_scaling


class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.pwd = r"/home/workspace/home/data"
        self.root_dir='/home/workspace/home/data'
        self.model_path = '/home/workspace/home/RESULTS/2023-12-17_1807_Basic_unet/model.pth'
        self.n_epochs = 1
        self.learning_rate = 0.0002
        self.batch_size = 64
        self.patch_size = 64
        self.test_results_dir = "RESULTS"
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 1 - self.train_ratio - self.val_ratio
        self.tensorboard_test_writer = SummaryWriter(comment="_test")

if __name__ == "__main__":
    # Get configuration

    # TASK: Fill in parameters of the Config class and specify directory where the data is stored and 
    # directory where results will go
    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)
    #print(data)
    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality
    data_size = len(data)
    keys = range(data_size)
    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    train_data = []
    train_size = int(c.train_ratio * data_size)
    val_data   = []
    val_size   = int(c.val_ratio * data_size)
    test_data  = []
    
    for element in keys:
        if len(train_data) < train_size:
           train_data.append(element)
        elif len(val_data) < val_size:
           val_data.append(element)
        else:
           test_data.append(element)             
    #debug 
    

    # Instantiate the model with the same architecture
    model = UNet(num_classes=3)
    # Load the saved parameters
    print('Load model....')
    model.load_state_dict(torch.load(c.model_path,map_location = torch.device('cpu')))

    # Set the model to evaluation mode (if only for inference)
    model.eval()
    inference_agent = UNetInferenceAgent(model=model, device='cpu')

    print("Testing...")
    dc_list = []
    jc_list = []    

    for  i in val_data:
         x=data[i]
         segmentation_mask_hat= inference_agent.single_volume_inference(x["image"])
         reference   = (x["seg"] > 0).astype(int)
         dc = Dice3d(segmentation_mask_hat, reference)
         jc = Jaccard3d(segmentation_mask_hat, reference)
         dc_list.append(dc)
         jc_list.append(jc)
         break   
    mean_dice = np.mean(dc_list)
    mean_jaccard = np.mean(jc_list)
    print(f'mean dice : {mean_dice}')
    print(f'mean jaccard : {mean_jaccard}')     




    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    #results_json = exp.run_test()

    #results_json["config"] = vars(c)

    #with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
    #    json.dump(results_json, out_file, indent=2, separators=(',', ': '))

