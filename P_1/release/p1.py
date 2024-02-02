import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):

    image = Image.open(filename)
    image = image.convert("RGB")
    image_np = np.array(image)
    image_float = image_np.astype(np.float32)/255.0

    return image_float

### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):

    ndims = np.ndim(img)

    image_h, image_w = img.shape[0], img.shape[1]
    filter_l, filter_k = filt.shape[0], filt.shape[1]
    
    padding_h = (filter_l - 1) // 2
    padding_w = (filter_k - 1) // 2

    # print("Image Shapes: m x n = ", image_h, image_w)
    # print("Image Dimensions: ", ndims)
    # print("Filter Shapes: k x k = ", filter_l, filter_k)

    if(ndims == 2):
        
        channels = 0
        convoluted_image = np.zeros((image_h, image_w))
        padded_image = np.pad(img, [(padding_h, padding_h), (padding_w, padding_w)], 'constant', constant_values=(0, 0))
        
        for row in range(padding_h, image_h + padding_h):
            for column in range(padding_w, image_w + padding_w):
                partial_image = padded_image[row - padding_h : row + padding_h + 1, column - padding_w : column + padding_w + 1]
                convoluted_image[row - padding_h][column - padding_w] = np.sum(partial_image * np.flip(filt))
    
    elif (ndims == 3):
        
        channels = 3
        convoluted_image = np.zeros((image_h, image_w, channels))
        
        for channel in range(channels):
            padded_image = np.pad(img[:,:,channel], [(padding_h, padding_h), (padding_w, padding_w)], 'constant', constant_values=(0, 0))

            for row in range(padding_h, image_h + padding_h):
                for column in range(padding_w, image_w + padding_w):
                    partial_image = padded_image[row - padding_h : row + padding_h + 1, column - padding_w : column + padding_w + 1]
                    convoluted_image[row - padding_h][column - padding_w][channel] = np.sum(partial_image * np.flip(filt))


    # print("Concoluted Image Dimensions: ", np.ndim(convoluted_image))
    # print("Concoluted Image Shapes: ", convoluted_image.shape[0], convoluted_image.shape[1], convoluted_image.shape[2])
    # print("Convoluted Image as array: ", np.asarray(convoluted_image))
    return convoluted_image



### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):

    # gaussian_filt = np.zeros((k, k))
    # sum = 0

    # for row in range(k):
    #     for column in range(k):
    #         gaussian_filt[row, column] = (( 1 / (2 * np.pi * (sigma**2) )) * np.exp(- ( ((row - (k//2))**2) + ((column - (k//2))**2) ) / (2*(sigma**2)) ))
    #         sum += gaussian_filt[row][column]

    # for row in range(k):
    #     for column in range(k):
    #         gaussian_filt[row, column] /= sum

    temp = np.arange(k)
    gaussian_filter = ((1/ np.sqrt(2*np.pi * (sigma**2)))) * np.exp(-((temp - (k//2))**2) / (2*(sigma**2)))
    gaussian_filter = np.expand_dims(gaussian_filter, axis = 1)
    gaussian_filter = gaussian_filter@gaussian_filter.T
    gaussian_filter /= np.sum(gaussian_filter)
    
    # print("Gaussing Filter Shape before norm : ", gaussian_filt.shape)
    # print("Gaussian Filt Sum: ", np.sum(gaussian_filt))
    # print("Gaussing Filter Norm Value: ", gaussian_filt)
    # print("Gaussing Filter Shape: ", gaussian_filt.shape)
    
    return gaussian_filter

### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):

    Y = 0.2125 * img[:, :, 0] + 0.7154 * img[:, :, 1] + 0.0721 * img[:, :, 2]
    filt_1 = gaussian_filter(5, 1)
    gaussian_filtered_1 = convolve(Y, filt_1)

    filt_2 = np.array([[0.5, 0, -0.5]])
    filt_3 = filt_2.T

    gaussian_filtered_2 = convolve(gaussian_filtered_1, filt_2)
    gaussian_filtered_3 = convolve(gaussian_filtered_1, filt_3)

    gradient_magnitude = np.sqrt((gaussian_filtered_2 ** 2) + (gaussian_filtered_3 ** 2))
    gradient_angle = np.arctan2(gaussian_filtered_3, gaussian_filtered_2)
    
    return gradient_magnitude, gradient_angle

##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):

    dist = np.abs((x * np.cos(theta)) + (y * np.sin(theta)) + c)
    
    return dist < thresh

### TODO 6: Write a function to draw a set of lines on the image. 
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs. 
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):
    
    img_2 = img.copy()
    lines = np.array(lines)
    
    for row_pixel in range(img.shape[1]):
        for column_pixel in range(img.shape[0]):
            pixel_trues = check_distance_from_line(row_pixel, column_pixel, lines[:, 0], lines[:, 1], thresh)
            if(True in pixel_trues):
                img_2[column_pixel][row_pixel] = np.array([[1, 0, 0]])

    return img_2

### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n), 
### as well as a set of possible theta values and a set of possible c values. 
### If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and** 
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    
    output_vote_array = np.zeros((len(thetas), len(cs)))

    for row_pixel in range(gradmag.shape[0]):
        for column_pixel in range(gradmag.shape[1]):
            if (gradmag[row_pixel][column_pixel] > thresh1):
                for theta_idx, theta in enumerate(thetas):
                    dist_boolean = check_distance_from_line(column_pixel, row_pixel, theta, cs, thresh2)
                    cs_idx = np.where(dist_boolean == True)
                    if(np.abs((theta - gradori[row_pixel][column_pixel])) < thresh3):
                        output_vote_array[theta_idx][cs_idx[0]] += 1


    # for row_pixel in range(gradmag.shape[0]):
    #     for column_pixel in range(gradmag.shape[1]):
    #         if (gradmag[row_pixel][column_pixel] > thresh1):
    #             for theta_idx, theta in enumerate(thetas):
    #                 dist_boolean = check_distance_from_line(row_pixel, column_pixel, theta, cs, thresh2)
    #                 for cs_idx, bool in enumerate(dist_boolean):
    #                     if(bool == True):
    #                         if((theta - gradori[row_pixel][column_pixel]) < thresh3):
    #                             output_vote_array[theta_idx][cs_idx] += 1
    
    # print(output_vote_array)
    # print(np.max(output_vote_array))
    # plt.imshow(output_vote_array)
    # plt.show()
    
    return output_vote_array

### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if: 
### (a) Its votes are greater than thresh, **and** 
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the 
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):

    local_max_votes = []

    theta_len = len(thetas)
    cs_len = len(cs)

    for theta_indx, theta in enumerate(thetas):
        for cs_indx, c in enumerate(cs):

            if (votes[theta_indx][cs_indx] > thresh):
                curr_vote_max = True

                vote_max = np.max(votes[max(0, theta_indx - nbhd) : min(theta_indx + nbhd + 1, theta_len), max(0, cs_indx - nbhd) : min(cs_indx, cs_indx + nbhd + 1, cs_len)])
                
                if (vote_max > votes[theta_indx][cs_indx]):
                    curr_vote_max = False
                    break
                if curr_vote_max:
                    local_max_votes.append((theta, c))

    return local_max_votes
    
# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines
   
    
