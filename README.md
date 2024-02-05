## Computer Vision Projects

### P_1 - Hough Transforms

We identify the edges around the given object and perform hough voting to determine the shapes of the object (lines in particular).

On the image - "same" convolution is applied using a gaussian filter to remove the noise present in it. The gradient magnitude and orientation of the image is calculated to determine the distance between a given point in the image to a line in parameter space. Once this is done, hough voting is performed to determine the highest number of votes that belongs to a particular line equation to determine the line(s) present within the given image.

##### Example 1
![Example 1](./P_1/release/img.jpg)

##### Results
|||||
|:--------:|:--------:|:--------:|:--------:|
|<img width='400' height='400' alt='Gaussian Filtering' src='./P_1/results/blurredImage_noiseRemoval.png'> <figcaption> Gaussian Filtering </figcaption>| <img width='400' height='400' alt='Image Gradients 1' src='./P_1/results/imageGradient.png'> <figcaption> Image Gradient (1/3) </figcaption>|<img width='400' height='400' alt='Image Gradients 2' src='./P_1/results/imageGradient.png'> <figcaption> Image Gradient (2/3) </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_1/results/imageGradient_2.png'> <figcaption> Image Gradient (3/3) </figcaption> |

|||
|:--------:|:--------:|
|<img width='400' height='400' alt='Image Gradients 2' src='./P_1/results/houghVoting.png'> <figcaption> Hough Voting </figcaption>| <img width='400' height='400' alt='Image Gradients 2' src='./P_1/results/localMaximasInHoughVoting.png'> <figcaption> Line Detection </figcaption> |

##### Example 2

|||
|:--------:|:--------:|
|<img width='400' height='400' alt='Image Gradients 2' src='./P_1/release/img2.png'> <figcaption> Example 2 </figcaption>|<img width='400' height='400' alt='Image Gradients 2' src='./P_1/results/example2.png'> <figcaption> Example 2 Line Detection </figcaption>|

The detailed project description can be found [here](./P_1/release/writeup.pdf).

### P_2 - Feature Detection and Matching  

##### Inputs
||||
|:--------:|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_2/sp23_release/resources/graf/img1.png'> <figcaption> Example 1 - Graffiti </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/sp23_release/resources/yosemite/yosemite1.jpg'> <figcaption> Example 2 - Yosemite </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/sp23_release/resources/triangle1.jpg'> <figcaption> Example 3 - Triangle </figcaption> |


##### Feature Detection

The features within the images are identified through `Harris Corner Point detection` method.  Here, for each pixel within the image - to calculate the strength and the orientation of the pixel - we consider a window of pixels around that point. We consider a `5x5` gaussian filter window with a `3x3` Sobel Operator to compute the `(x, y)` derivatives.

The strongest keypoints which are local maxima within a `7x7` window size are selected.

||||
|:--------:|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example1_featureDetection.jpg'> <figcaption> Example 1 - Graffiti </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example2_featureDetection.jpg'> <figcaption> Example 2 - Yosemite </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example3_featureDetection.jpg'> <figcaption> Example 3 - Triangle </figcaption> |

##### Feature Description

Now that the features of interest are identified within the images, these features are then assigned descriptors centered at each interest point. This descriptor will be the representation that will be used to compare features in different images to see if they match.

Two feature descriptors are created - **SimpleFeatureDescriptor** and a simplified **Multi-Scale Oriented Patches (MOPS)**

The **SimpleFeatureDescriptor** considers the pixel intensity values in a `5x5` neighbourhood window for each pixel to generate a descriptor.

In **MOPS** - a `8x8` oriented patch sub-sampled from a `40×40` pixel region around the feature is computed. This is acheived through a transformation matrix which transforms the `40×40` rotated window around the feature to an `8×8` patch rotated so that its keypoint orientation points to the right. We also normalize the patch to have `zero mean` and `unit variance`.

`cv2.wrapAffine` is used to perform the transformation through a sequence of `translation (T1), rotation (R), scaling (S) and translation (T2)`.  

Left- multiplied transformations are combined right-to-left so the transformation matrix is the matrix product `T2 S R T1`.

##### Feature Matching

As the features of interest are detected and are assigned descriptors - they can now be matched between 2 images to identify same pixels in 2 images of varying linear transformations and orientations.

Two features are compared and the scalar distance between them is calculated. The smallest feature distance results in the best match.

Distance functions implemented:
1. `Sum of Squared Differences (SSD)`
2. `Ratio Test`

||||
|:--------:|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example_1_featureMatching.jpg'> <figcaption> Example 1 - Graffiti </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example_2_featureMatching.jpg'> <figcaption> Example 2 - Yosemite </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_2/results/example_3_featureMatching_ratioTest.jpg'> <figcaption> Example 3 - Triangle </figcaption> |

The detailed project description can be found [here](./P_2/sp23_release/CS4_5670_PA2_WriteUp.pdf).


### P_3 - Photometric and Plane Sweep Stereo

#### 1. Photometric Stereo

From a stack of images taken from the same viewpoint under different, known illumination directions - the albedo and the normals of the object surfaces are recovered.

##### Inputs

1: A tentacle - which is a 3D mesh - that is rendered under 9 different directional illumination settings.

2: A Miniature Cat

##### Object Albedos

`Albedo` is defined to be the measure of the propotion of the incident light or radiation that is reflected by a surface.

|||
|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/tentacle_albedo.png'> <figcaption> Tentacle's Albedo </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/cat_albedo.png'> <figcaption> Cat's Albedo </figcaption> |

##### Object Normals

Here, the color channels - Red indicates the normal is pointing to the right (+x direction), Green indicates the normal is pointing up (+y direction) and Blue indicates the normal is pointing out of the screen (+z direction).

|||
|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/tentacle_normals.png'> <figcaption> Tentacle's Normals </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/cat_normals.png'> <figcaption> Cat's Normals </figcaption> |

#### 2. Plane Sweep Stereo

Given two calibrated images of the same scene, that are taken from different viewpoints, a rough depth map is recovered.

Here, stereo block matching is performed. Given two images, a window is slid along the epipolar line and the contents of that window is compared with the reference window in the left image.

The cost similarity measure used is the `Normalized Cross Correlation (NCC)`. The below images identifies the argmax depth according to the NCC cost volume. White is near and black is far.

|||
|:--------:|:--------:|
| <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/tentacle_ncc.png'> <figcaption> Tentacle's NCC </figcaption> | <img width='400' height='400' alt='Image Gradients 2' src='./P_3/CV-PA3-Outputs/Flowers_ncc.png'> <figcaption> Flower's NCC </figcaption> |

#### 3. Depth Map Estimation

Given the normal map and/or depth map - a mesh is reconstructed.

The results can be found [here](./P_3/CV-PA3-Outputs/part-3-depth-map-reconstruction.pdf).

The detailed project description can be found [here](./P_3/release/writeup.pdf).

### P_4 - Convolutional Neural Networks (CNNs) for Image Classification


The image classification results can be found [here](./P_4/results/Project4-WriteUp.pdf).

The detailed project description can be found [here](./P_4/Release/PA4_prompt.pdf).