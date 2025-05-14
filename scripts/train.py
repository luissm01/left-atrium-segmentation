#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Finally we are going to train our atrium segmentation network

# ## Imports:
# 
# * pathlib for easy path handling
# * torch for tensor handling
# * pytorch lightning for efficient and easy training implementation
# * ModelCheckpoint and TensorboardLogger for checkpoint saving and logging
# * imgaug for Data Augmentation
# * numpy for file loading and array ops
# * matplotlib for visualizing some images
# * Our dataset and model
# 
# 

# In[1]:


from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import imgaug.augmenters as iaa
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("../scripts/")
from dataset import CardiacDataset
from model import UNet


# ## Dataset Creation
# We begin by defining the data augmentation pipeline for the training dataset. This includes random affine transformations such as scaling (zoom in/out) and rotation, as well as elastic deformations to improve generalization and robustness.
# 

# In[4]:


seq = iaa.Sequential([
    iaa.Affine(scale=(0.85, 1.15),
              rotate=(-45, 45)),
    iaa.ElasticTransformation()
])


# Next, we instantiate the training and validation datasets using the `CardiacDataset` class. The training set receives the augmentation sequence, while the validation set remains unchanged.

# In[5]:


# Create the dataset objects
train_path = Path("../data/Preprocessed/train/")
val_path = Path("../data/Preprocessed/val")

train_dataset = CardiacDataset(train_path, seq)
val_dataset = CardiacDataset(val_path, None)




# 
# We print out the total number of 2D slices used for training and validation, which represent the total number of individual axial images extracted from all subjects.
# 
# To verify the slice distribution, we iterate through each subject folder and count how many 2D slices were preprocessed and stored in each `slices/` directory. This gives us a detailed view of how many samples are contributed by each subject.
# 
# This step is useful to ensure the dataset was correctly preprocessed and to check for potential imbalances between subjects.

# We now create the training and validation dataloaders using a batch size of 8 and 4 worker processes. The training data is shuffled to improve generalization, while the validation data is kept in order.

# In[ ]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False


# In[ ]:


# Use more workers if safe (e.g. script, not notebook)
if os.name == 'nt' and not is_notebook():
    num_workers = min(8, os.cpu_count())  # Use up to 8 workers safely
    persistent_workers = True
else:
    num_workers = 0
    persistent_workers = False


# In[9]:


batch_size = 8

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)


# ## Custom Loss
# In medical image segmentation, the goal is to predict a mask that matches as closely as possible the true shape of the organ or structure. For this, we need a loss function that accurately measures how similar two binary masks are: the predicted one and the ground truth.
# 
# ### Why not use Cross-Entropy?
# 
# Cross-Entropy is commonly used in classification tasks, but it treats each pixel independently. In segmentation, this often leads to poor performance when the object of interest (like the left atrium) occupies only a small part of the image — a situation known as **class imbalance**. The model might learn to predict "background" everywhere and still get a good loss.
# 
# ### Why use Dice Loss?
# 
# Dice Loss is specifically designed to measure the **overlap between two binary masks**. It focuses directly on the match between the predicted mask and the ground truth.
# 
# The Dice Loss is based on the **Dice coefficient**, a metric from the field of set similarity:
# 
# $$
# L(\hat{y}, y) = 1 - \frac{2 \cdot |\hat{y} \cap y|}{|\hat{y}| + |y|}
# $$
# 
# Where:
# - $\hat{y}$ is the predicted mask.
# - $y$ is the ground truth mask.
# - $\hat{y} \cap y$ is the number of pixels where both are 1 (true positives).
# - $|\hat{y}|$ is the number of predicted foreground pixels.
# - $|y|$ is the number of ground truth foreground pixels.
# 
# The Dice Loss ranges from 0 (perfect match) to 1 (no overlap). Since our goal is to maximize similarity, we minimize this loss.
# 
# ### Why is it better for segmentation?
# 
# Dice Loss directly optimizes for **overlap**, which is exactly what we care about in segmentation. It is:
# - Robust to class imbalance.
# - Easy to compute.
# - Closely aligned with how we evaluate segmentation quality in practice.
# 
# This makes it a strong choice for training models to segment small, complex structures like the left atrium.

# In[10]:


class DiceLoss(torch.nn.Module):
    """
    class to compute the Dice Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
                
        # Flatten label and prediction tensors
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)
        counter = (pred * mask).sum()  # Numerator       
        denum = pred.sum() + mask.sum() + 1e-8  # Denominator. Add a small number to prevent NANS
        dice =  (2*counter)/denum
        return 1 - dice


# ## Full Segmentation Model
# 
# This class wraps the full segmentation pipeline into a single PyTorch Lightning module.  
# It includes:
# 
# - The U-Net architecture for left atrium segmentation.
# - Dice Loss as the training objective, optimized for overlap-based evaluation.
# - Adam optimizer with a fixed learning rate.
# - Training and validation steps with automatic logging.
# - Visual logging of predictions using TensorBoard every few batches.
# 
# The forward pass produces the predicted mask using a sigmoid activation, while training and validation steps compute the loss and optionally visualize predictions. PyTorch Lightning handles the optimization and backpropagation internally based on the returned loss.

# In[26]:


class AtriumSegmentation(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = UNet()  # Define the segmentation model (U-Net)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)  # Use Adam optimizer
        self.loss_fn = DiceLoss()  # Use Dice Loss for segmentation

    def forward(self, data):
        # Forward pass through the network with sigmoid activation
        return torch.sigmoid(self.model(data))

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        mri, mask = batch
        mask = mask.float()  # Convert mask to float for Dice calculation

        # Forward pass
        pred = self(mri)

        # Compute Dice Loss
        loss = self.loss_fn(pred, mask)

        # Log training loss to TensorBoard
        self.log("Train Dice", loss)

        # Log sample images every 50 steps
        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Train")

        return loss  # Lightning will handle backward and optimization

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        mri, mask = batch
        mask = mask.float()

        # Forward pass
        pred = self(mri)

        # Compute Dice Loss
        loss = self.loss_fn(pred, mask)

        # Log validation loss
        self.log("Val Dice", loss)

        # Log images every 2 validation steps
        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), "Val")

        return loss

    def log_images(self, mri, pred, mask, name):
    	# Only log one image per epoch per mode
    	if self.global_step % self.trainer.num_training_batches != 0:
        	return

    	pred = pred > 0.5

    	fig, axis = plt.subplots(1, 2)

    	# Ground truth
    	axis[0].imshow(mri[0][0], cmap="bone")
    	mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
    	axis[0].imshow(mask_, alpha=0.6)
    	axis[0].set_title("Ground Truth")

    	# Prediction
    	axis[1].imshow(mri[0][0], cmap="bone")
    	mask_ = np.ma.masked_where(pred[0][0] == 0, pred[0][0])
    	axis[1].imshow(mask_, alpha=0.6)
    	axis[1].set_title("Prediction")

    	self.logger.experiment.add_figure(name, fig, self.global_step)
    	plt.close(fig)

    def configure_optimizers(self):
        	# Return the optimizer to be used during training
        	return [self.optimizer]


# In[27]:

if __name__ == "__main__":
	# Instanciate the model and set the random seed
	torch.manual_seed(0)
	model = AtriumSegmentation()

	# Create the checkpoint callback
	checkpoint_callback = ModelCheckpoint(
    		monitor='Val Dice',
    		save_top_k=10,
    		mode='min')
	from pytorch_lightning.callbacks import EarlyStopping

	early_stop_callback = EarlyStopping(
    		monitor='Val Dice',
    		patience=30,       # N° de epochs sin mejora
    		mode='min'         # Porque el DiceLoss es menor = mejor
	)


	gpus = 1 #TODO
	trainer = pl.Trainer(gpus=gpus, logger=TensorBoardLogger(save_dir="../logs"), log_every_n_steps=1,
                     callbacks=[checkpoint_callback, early_stop_callback],max_epochs=75)

	trainer.fit(model, train_loader, val_loader)
