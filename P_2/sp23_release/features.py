import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial
# import matplotlib.pyplot as plt
import transformations

## Helper functions ############################################################

def inbounds(shape, indices):
    '''
        Input:
            shape -- int tuple containing the shape of the array
            indices -- int list containing the indices we are trying 
                       to access within the array
        Output:
            True/False, depending on whether the indices are within the bounds of 
            the array with the given shape
    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


## Keypoint detectors ##########################################################

class KeypointDetector(object):
    # Implement in child classes
    def detectKeypoints(self, image):
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):
    '''
        Compute silly example features. This doesn't do anything meaningful, but
        may be useful to use as an example.
    '''

    def detectKeypoints(self, image):
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):

    def computeHarrisValues(self, srcImage):
        '''
        Input:
            srcImage -- Grayscale input image in a numpy array with
                        values in [0, 1]. The dimensions are (rows, cols).
        Output:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
            orientationImage -- numpy array containing the orientation of the
                                gradient at each pixel in degrees.
        '''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # print(harrisImage.shape)
        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'. Also compute an 
        # orientation for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN

        gaussianWindowSize = 5

        gaussian_array = np.zeros((gaussianWindowSize, gaussianWindowSize))
        gaussian_array[gaussianWindowSize // 2][gaussianWindowSize // 2] = 1
        gaussian_filt = ndimage.gaussian_filter(gaussian_array, sigma = 0.5)
       
        harris_matrix = np.zeros((2, 2))

        # Calculating and padding the derivatives

        Ix = ndimage.sobel(srcImage, axis = 1)
        Iy = ndimage.sobel(srcImage, axis = 0)

        padding_h = (gaussianWindowSize - 1) // 2
        padding_w = (gaussianWindowSize - 1) // 2

        Ix_pad = np.pad(Ix, (padding_h, padding_w), mode = 'edge')
        Iy_pad = np.pad(Iy, (padding_h, padding_w), mode = 'edge')
        
        Ix_2 = Ix_pad * Ix_pad
        Iy_2 = Iy_pad * Iy_pad
        Ix_Iy = Ix_pad * Iy_pad

        for row_pixel in range(height):
            for column_pixel in range(width):

                harris_matrix[0, 0] = np.sum(gaussian_filt * Ix_2[row_pixel: (row_pixel + gaussianWindowSize), column_pixel: (column_pixel + gaussianWindowSize)])
                harris_matrix[0, 1] = np.sum(gaussian_filt * Ix_Iy[row_pixel: (row_pixel + gaussianWindowSize), column_pixel: (column_pixel + gaussianWindowSize)])
                harris_matrix[1, 0] = np.sum(gaussian_filt * Ix_Iy[row_pixel: (row_pixel + gaussianWindowSize), column_pixel: (column_pixel + gaussianWindowSize)])
                harris_matrix[1, 1] = np.sum(gaussian_filt * Iy_2[row_pixel: (row_pixel + gaussianWindowSize), column_pixel: (column_pixel + gaussianWindowSize)])

                harrisImage[row_pixel, column_pixel] = np.linalg.det(harris_matrix) - (0.1*(np.matrix.trace(harris_matrix))**2)
                orientationImage[row_pixel, column_pixel] = np.rad2deg(np.arctan2(Iy[row_pixel, column_pixel], Ix[row_pixel, column_pixel]))

        # TODO-BLOCK-END
        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
        Input:
            harrisImage -- numpy array containing the Harris score at
                           each pixel.
        Output:
            destImage -- numpy array containing True/False at
                         each pixel, depending on whether
                         the pixel value is the local maxima in
                         its 7x7 neighborhood.
        '''

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN

        destImage = np.zeros((harrisImage.shape), dtype= bool)

        localMaximaWindowSize = 7
        harrisImage_pad = harrisImage
        harrisImage_filtered = ndimage.maximum_filter(harrisImage_pad, size = (localMaximaWindowSize, localMaximaWindowSize))
        
        for row_pixel in range(destImage.shape[0]):
            for column_pixel in range(destImage.shape[1]):
                if (harrisImage_filtered[row_pixel, column_pixel] == harrisImage[row_pixel, column_pixel]):
                    destImage[row_pixel, column_pixel] = True

        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):
        '''
        Input:
            image -- BGR image with values between [0, 255]
        Output:
            list of detected keypoints, fill the cv2.KeyPoint objects with the
            coordinates of the detected keypoints, the angle of the gradient
            (in degrees), the detector response (Harris score for Harris detector)
            and set the size to 10.
        '''
        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        # Create grayscale image used for Harris detection
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # computeHarrisValues() computes the harris score at each pixel
        # position, storing the result in harrisImage.
        # You will need to implement this function.
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)

        # Compute local maxima in the Harris image.
        # You will need to implement this function.
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        # Loop through feature points in harrisMaxImage and fill in information
        # needed for descriptor computation for each point.
        # You need to fill x, y, and angle.
        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()

                # TODO 3: Fill in feature f with location and orientation
                # data here. Set f.size to 10, f.pt to the (x,y) coordinate,
                # f.angle to the orientation in degrees and f.response to
                # the Harris score
                # TODO-BLOCK-BEGIN

                f.pt = (x, y)
                # Dummy size
                f.size = 10
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]

                features.append(f)

                # TODO-BLOCK-END
        print("Length of features: ", len(features))
        return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        detector = cv2.ORB_create()
        return detector.detect(image)




## Feature descriptors #########################################################

class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
                         descriptors at the specified coordinates
        Output:
            desc -- K x 25 numpy array, where K is the number of keypoints
        '''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        height = grayImage.shape[0]
        width = grayImage.shape[1]

        # print("Height of the grayImage : ", height)
        # print("Width of the grayImage : ", width)
        # print("Descriptor shapes: ", desc.shape)

        for indx, feature in enumerate(keypoints):
            x, y = int(feature.pt[0]), int(feature.pt[1])

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN

            descriptorWindowHeight = 5
            descriptorWindowWidth = 5
            descriptorWindow = np.zeros((descriptorWindowHeight, descriptorWindowWidth))

            for i in range(descriptorWindowHeight):
                for j in range(descriptorWindowWidth):
                    if ((((i + y - 2) >= 0 and (i + y - 2) < height)) and (((j + x - 2) >= 0 and (j + x - 2) < width))):
                        descriptorWindow[i, j] = grayImage[i + y - 2, j + x - 2]

            desc[indx] = np.reshape(descriptorWindow, (1, 25))

            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
        Input:
            image -- BGR image with values between [0, 255]
            keypoints -- the detected features, we have to compute the feature
            descriptors at the specified coordinates
        Output:
            desc -- K x W^2 numpy array, where K is the number of keypoints
                    and W is the window size
        '''
        image = image.astype(np.float32)
        image /= 255.
        # This image represents the window around the feature you need to
        # compute to store as the feature descriptor (row-major)
        gaussianWindowSize = 8
        desc = np.zeros((len(keypoints), gaussianWindowSize * gaussianWindowSize))
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        grayImage = ndimage.gaussian_filter(grayImage, 0.5)

        # plt.imshow(grayImage)
        # plt.show()
        # quit()

        for indx, feature in enumerate(keypoints):
            # print(feature.pt)
            transMx = np.zeros((2, 3))

            # TODO 5: Compute the transform as described by the feature
            # location/orientation and store in 'transMx.' You will need
            # to compute the transform from each pixel in the 40x40 rotated
            # window surrounding the feature to the appropriate pixels in
            # the 8x8 feature descriptor image. 'transformations.py' has
            # helper functions that might be useful
            # Note: use grayImage to compute features on, not the input image
            # TODO-BLOCK-BEGIN
            x, y = int(feature.pt[0]), int(feature.pt[1])

            theta = np.deg2rad(feature.angle)
            theta_x = 0
            theta_y = 0
            theta_z = theta

            translationOneMx = transformations.get_trans_mx(np.array([-x, -y, 0]))
            rotationMx = transformations.get_rot_mx(theta_x, theta_y, -theta_z)
            scaleMx = transformations.get_scale_mx(1/5, 1/5, 1)
            translationTwoMx = transformations.get_trans_mx(np.array([4, 4, 0]))

            transMx = translationTwoMx @ scaleMx @ rotationMx @ translationOneMx
            transMx = np.array([[transMx[0][0], transMx[0][1], transMx[0][3]],
                                [transMx[1][0], transMx[1][1], transMx[1][3]]])

            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

            # Call the warp affine function to do the mapping
            # It expects a 2x3 matrix
            destImage = cv2.warpAffine(grayImage, transMx, (gaussianWindowSize, gaussianWindowSize), flags=cv2.INTER_LINEAR)

            # TODO 6: Normalize the descriptor to have zero mean and unit
            # variance. If the variance is negligibly small (which we
            # define as less than 1e-10) then set the descriptor
            # vector to zero. Lastly, write the vector to desc.
            # TODO-BLOCK-BEGIN

            mean, std = cv2.meanStdDev(destImage)

            descriptor = cv2.subtract(destImage, mean)
            descriptor = cv2.divide(descriptor, std)

            if (std**2) < 1e-10:
                descriptor = np.zeros_like(descriptor)

            descriptor = np.reshape(descriptor, (1, 64))
            desc[indx] = descriptor
            
            # raise Exception("TODO in features.py not implemented")
            # TODO-BLOCK-END

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        descriptor = cv2.ORB_create()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc




## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        raise NotImplementedError

    # Evaluate a match using a ground truth homography. This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6]*x + h[7]*y + h[8]

        return np.array([(h[0]*x + h[1]*y + h[2]) / d,
            (h[3]*x + h[4]*y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The distance between the two features
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 7: Perform simple feature matching. This uses the SSD
        # distance between two feature vectors, and matches a feature in
        # the first image with the closest feature in the second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN

        euclideanDistanceBetweenDescriptors = spatial.distance.cdist(desc1, desc2, metric = 'sqeuclidean')

        for features in range(euclideanDistanceBetweenDescriptors.shape[0]):

            featureMatch = cv2.DMatch()
            featureMatch.queryIdx = features
            featureMatch.trainIdx = np.argmin(euclideanDistanceBetweenDescriptors[features])
            featureMatch.distance = euclideanDistanceBetweenDescriptors[featureMatch.queryIdx][featureMatch.trainIdx]

            matches.append(featureMatch)
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
        Input:
            desc1 -- the feature descriptors of image 1 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
            desc2 -- the feature descriptors of image 2 stored in a numpy array,
                dimensions: rows (number of key points) x
                columns (dimension of the feature descriptor)
        Output:
            features matches: a list of cv2.DMatch objects
                How to set attributes:
                    queryIdx: The index of the feature in the first image
                    trainIdx: The index of the feature in the second image
                    distance: The ratio test score
        '''
        matches = []
        assert desc1.ndim == 2
        assert desc2.ndim == 2
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image. If the SSD distance is negligibly small, in this case less
        # than 1e-5, then set the distance to 1. If there are less than two features,
        # set the distance to 0.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")

        euclideanDistanceBetweenDescriptors = spatial.distance.cdist(desc1, desc2, metric = 'sqeuclidean')

        for features in range(euclideanDistanceBetweenDescriptors.shape[0]):

            sortedEuclideanDistance = sorted(euclideanDistanceBetweenDescriptors[features])
            firstSmallestFeatureDistance = sortedEuclideanDistance[0]
            
            featureMatch = cv2.DMatch()

            if(len(euclideanDistanceBetweenDescriptors[features]) < 2):
                featureMatch.queryIdx = features
                featureMatch.trainIdx = int(np.where(euclideanDistanceBetweenDescriptors[features] == firstSmallestFeatureDistance)[0])
                featureMatch.distance = 0
            else:
                secondSmallestFeatureDistance = sortedEuclideanDistance[1]
                
                if (secondSmallestFeatureDistance) < 1e-5:
                    ratioOfFeatures = 1
                else:
                    ratioOfFeatures = firstSmallestFeatureDistance / secondSmallestFeatureDistance
                    
                featureMatch.queryIdx = features
                featureMatch.trainIdx = int(np.where(euclideanDistanceBetweenDescriptors[features] == firstSmallestFeatureDistance)[0])
                featureMatch.distance = ratioOfFeatures

            matches.append(featureMatch)
        # TODO-BLOCK-END

        return matches