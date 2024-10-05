import unittest
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'PyTorchLayerViz')))

from main import get_feature_maps

pretrained_model = models.vgg16(pretrained=True)
input_image_path = 'brain.tif'

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((256, 256)),            
    transforms.CenterCrop(224),
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])



class TestGetFeatureMaps(unittest.TestCase):
    def test_maxPool(self):
        layers_to_check= [nn.MaxPool2d]
        numpyArr_from_function = get_feature_maps(model = pretrained_model, layers_to_check = layers_to_check, input_image_path = input_image_path)

        numpyArr_from_file = np.load('output_images/test_maxPool.npy')

        np.testing.assert_array_equal(numpyArr_from_function, numpyArr_from_file)
    def test_moreLayers(self):
        layers_to_check= [nn.MaxPool2d, nn.Conv2d]
        numpyArr_from_function = get_feature_maps(model = pretrained_model, layers_to_check = layers_to_check, input_image_path = input_image_path)

        numpyArr_from_file = np.load('output_images/test_moreLayers.npy')

        np.testing.assert_array_equal(numpyArr_from_function, numpyArr_from_file)   
    def test_transform(self):
        layers_to_check= [nn.MaxPool2d, nn.Conv2d]
        numpyArr_from_function = get_feature_maps(model = pretrained_model, layers_to_check = layers_to_check, input_image_path = input_image_path, transform = transform)

        numpyArr_from_file = np.load('output_images/test_transform.npy')

        np.testing.assert_array_equal(numpyArr_from_function, numpyArr_from_file)     

if __name__ == '__main__':
    unittest.main()
