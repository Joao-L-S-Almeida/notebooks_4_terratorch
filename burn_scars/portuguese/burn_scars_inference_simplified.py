#!/usr/bin/env python
# coding: utf-8

# In[6]:


import argparse
import os
from typing import List, Union
import re
import datetime
import numpy as np
import rasterio
import torch
import rioxarray
import yaml
from einops import rearrange
from terratorch.cli_tools import LightningInferenceModel
from terratorch.utils import view_api
from terratorch.datasets import HLSBands

# In[16]:


output_dir = "inference_output"
config_file = "burn_scars_config_tiled.yaml"
checkpoint = "checkpoints/Prithvi_EO_V2_300M_BurnScars.pt"
input_dir = "data/examples/"
example_file = "data/examples/subsetted_512x512_HLS.S30.T10SEH.2018190.v1.4_merged.tif"
input_indices = [0,1,2,3,4,5]
img_size = 512
predict_dataset_bands=[
      "BLUE",
      "GREEN",
      "RED",
      "NIR_NARROW",
      "SWIR_1",
      "SWIR_2",
  ]
predict_output_bands = predict_dataset_bands

# In[13]:


os.makedirs(output_dir, exist_ok=True)


# In[18]:


with open(config_file, "r") as f:
    config_dict = yaml.safe_load(f)


# In[22]:


lightning_model = LightningInferenceModel.from_config(config_file, checkpoint, predict_dataset_bands, predict_output_bands)


# In[20]:


predictions = lightning_model.inference_on_dir(input_dir)
print(predictions)


# In[ ]:




