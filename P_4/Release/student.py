# Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

# END IMPORTS

#########################################################
###              BASELINE MODEL
#########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 12, 3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(12, 24, 3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.fc = nn.Linear(1536, 128)
        self.relu4 = nn.ReLU()
        self.cls = nn.Linear(128, 16)
        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc(torch.reshape(x, (-1, 1536)))
        x = self.relu4(x)
        x = self.cls(x)
        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Forward pass
    # TODO-BLOCK-BEGIN
    running_loss   = 0.0
    num_correct    = 0.0
    total_images   = 0.0
    # Propagate batch through network
    outputs = net(inputs)
    # Calculate loss
    loss = criterion(outputs, labels.squeeze())
    # Prediction is class with highest class score
    _, preds = torch.max(outputs, 1)
    running_loss += loss.item()
    num_correct += torch.sum(preds == labels.data.reshape(-1))
    total_images += labels.data.numpy().size
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    # Zero out the parameter gradients
    optimizer.zero_grad()
    # Backpropagation
    loss.backward()
    # Take a step to update the network
    optimizer.step()
    # TODO-BLOCK-END

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        # Moving channels dimension from first dimension to last dimension of image.
        image = np.moveaxis(image, 0, -1)
        # Random x and y amount to shift image by.
        x = np.random.randint(-self.max_shift, self.max_shift + 1)
        y = np.random.randint(-self.max_shift, self.max_shift + 1)
        # Defining translation matrix.
        t_matrix = np.float32([[1, 0, x], [0, 1, y]])   
        # Applying transformation.
        image = cv2.warpAffine(image, t_matrix, (W, H)) 
        # Moving channels dimension back to first dimension of image.
        image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs: 
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN
        # Selecting 'contrast_factor' by sampling from uniform distribution.
        contrast_factor = np.random.uniform(self.min_contrast, self.max_contrast)
        # Getting the mean value for each of our three channels.
        channel_means = np.mean(image.reshape(3, -1), axis=1)
        # Creating the mean image given the channel means.
        mean_image = np.repeat(channel_means, H * W).reshape(3, H, W)
        # Calculating the image you get from subtracting the 'mean_image' from the original image.
        difference_image = image - mean_image
        # Adding back a scaled version of "difference_image" to "mean_image" to get the new image.
        image = mean_image + contrast_factor * difference_image
        # Clipping values that are less than 0 or greater than 1.
        image = np.clip(image, 0, 1)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        # Moving channels dimension from first dimension to last dimension of image.
        image = np.moveaxis(image, 0, -1)
        # Selecting 'rotation_angle' by sampling from uniform distribution.
        r_angle = np.random.uniform(-self.max_angle, self.max_angle)
        # Getting rotation matrix.
        r_matrix = cv2.getRotationMatrix2D(center=(W // 2, H // 2), angle=r_angle, scale=1)  
        # Applying transformation.
        image = cv2.warpAffine(image, r_matrix, (W, H))
        # Moving channels dimension back to first dimension of image.
        image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        # Getting random probability value.
        other_p = np.random.uniform(0, 1)
        # Horizontally flipping image if the probability we just generated is less than 'self.p'.
        if other_p < self.p:
            # Moving channels dimension from first dimension to last dimension of image.
            image = np.moveaxis(image, 0, -1)
            # Defining horizontal flip matrix.
            #
            # See this post for why we had to add 'W - 1' to our standard horizontal flip matrix...
            # https://stackoverflow.com/a/57864513.
            f_matrix = np.float32([[-1, 0, W - 1], [0, 1, 0]])   
            # Applying transformation.
            image = cv2.warpAffine(image, f_matrix, (W, H)) 
            # Moving channels dimension back to first dimension of image.
            image = np.moveaxis(image, -1, 0)
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN

    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN
     # Calculate loss.
    loss = criterion(output, label)
    # Calculating gradient of the loss with respect to the image.
    G = torch.autograd.grad(loss, img)
    # Deriving tensor of -1s and 1s, where elements that are -1 correspond to elements of 'G' that were negative.
    # and elements that are 1 correspond to elements of 'G that were positive.
    signs = G[0] / torch.abs(G[0])
    # Replacing all nan values with 1.
    signs = torch.nan_to_num(signs, nan=1.0)
    # Creating matrix a of perturbations.
    noise = signs * epsilon
    # Adding noise to image to get perturbed image.
    perturbed_image = img + noise
    # Clamping 'perturbed_image' values to be between 0 and 1.
    perturbed_image = torch.clamp(perturbed_image, min=0.0, max=1.0)
    # TODO-BLOCK-END

    return perturbed_image, noise

