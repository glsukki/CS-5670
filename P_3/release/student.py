# Please place imports here.
# BEGIN IMPORTS
import numpy as np
import scipy
# END IMPORTS


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- 3 x N array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 height x width x channels image with dimensions
                  matching the input images.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    # Converting list of images (where each image is of dimensions height x width x # of channels)
    # to since numpy array of dimensions # of images x (height * width) x # of channels.
    I_total = [image.reshape(image.shape[0] * image.shape[1], image.shape[2]) for image in images]
    I_total = np.array(I_total)

    # Array that will hold the calculated albedo values for all color channels.
    # Each element in the array will be the result for a particular color channel, so in the case of an RGB
    # image, this array will hold three arrays of shape (height, width), where the first array is the result
    # for R-channel, second array is the result for G-channel, and third array is result for B-channel.
    albedos_total = []
    # Will hold the sum of all the surface normal estimates (one estimate for each channel).
    surface_normals_total = None

    # If there are more than 0 images, calculating the albedos and 
    if len(images) > 0:
        for channel_index in range(images[0].shape[2]):
            # Getting image radiance matrix for current channel we are processing.
            I = I_total[:, :, channel_index]
            # Solving for G using the closed-form solution.
            G = (np.linalg.inv(lights @ lights.T) @ lights) @ I
            # Calculating norm of each column in 'G'. 
            G_norm = np.linalg.norm(G, axis=0)
            # Replacing norm values less than 1e-7 with 0. 
            G_norm[G_norm < 1e-7] = 0
            # Getting surface normal vectors by dividing each column of matrix G by its norm. 
            #
            # Note: If a value in 'G_norm' is 0, we simply set the result to 0, rather than dividing by 0.
            # Credit to Pranit Sharma for his elegant solution to this, found here: https://www.includehelp.com/python/numpy-how-to-return-0-with-divide-by-zero.aspx.
            surface_normals = np.divide(G, G_norm, where=G_norm!=0)
            # Updating 'surface_normals_total'.
            if surface_normals_total is None:
                surface_normals_total = surface_normals
            else:
                surface_normals_total = surface_normals_total + surface_normals
            # Reshaping albedos to dimensions height x width. 
            albedos = G_norm.reshape(images[0].shape[0], images[0].shape[1])
            # Adding to 'albedos_total' array. 
            albedos_total.append(albedos)

        # Combining different channels back together into one single image.
        albedo_result = np.dstack(albedos_total)
        # Dividing 'surface_normals_total' by the number of channels to get an average estimate. 
        surface_normals_result = surface_normals_total / images[0].shape[2]
        # Reshaping 'surface normals_result' to dimensions height x width x 3.
        surface_normals_result = surface_normals_result.T.reshape(images[0].shape[0], images[0].shape[1], 3)

    return albedo_result, surface_normals_result

def pyrdown_impl(image):
    """
    Prefilters an image with a gaussian kernel and then downsamples the result
    by a factor of 2.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/16 [ 1 4 6 4 1 ]

    Functions such as cv2.GaussianBlur and scipy.ndimage.gaussian_filter are
    prohibited.  You must implement the separable kernel.  However, you may
    use functions such as cv2.filter2D or scipy.ndimage.correlate to do the actual
    correlation / convolution. Note that for images with one channel, cv2.filter2D
    will discard the channel dimension so add it back in.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Downsampling should take the even-numbered coordinates with coordinates
    starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        down -- ceil(height/2) x ceil(width/2) x channels image of type
                float32.
    """
    # Creating 1D convolution kernel.
    kernel = np.array([[1, 4, 6, 4, 1]]) / 16
    # Array that will hold the result of the pre-filtering using the gaussian kernel.
    # Each element in the array will be the result for a particular color channel, so in the case of an RGB
    # image, this array will hold three arrays of shape (height, width), where the first array is the result
    # for R-channel, second array is the result for G-channel, and third array is result for B-channel.
    convolve_result = []
    # Iterating over the indices of all the channels in the image.
    for channel_index in range(image.shape[2]):
        # Grabbing the pixels corresponding only to the channel we are currently processing.
        sub_image = image[:, :, channel_index]
        # Applying 1D convolution kernel along x direction.
        sub_image = scipy.ndimage.correlate(sub_image, kernel, mode='mirror')
        # Applying 1D convolution kernel along y direction.
        sub_image = scipy.ndimage.correlate(sub_image, kernel.T, mode='mirror')
        # Adding result to 'convolve_result'.
        convolve_result.append(sub_image)
    # Combining different channels back together into one single image.
    prefiltering_result = np.dstack(convolve_result)
    # Keeping only the even-numbered coordinates (with coordinates starting at 0).
    downsampled_result = prefiltering_result[::2, ::2, :]
    return downsampled_result


def pyrup_impl(image):
    """
    Upsamples an image by a factor of 2 and then uses a gaussian kernel as a
    reconstruction filter.

    The following 1D convolution kernel should be used in both the x and y
    directions.
    K = 1/8 [ 1 4 6 4 1 ]
    Note: 1/8 is not a mistake.  The additional factor of 4 (applying this 1D
    kernel twice) scales the solution according to the 2x2 upsampling factor.

    Filtering should mirror the input image across the border.
    For scipy this is mode = mirror.
    For cv2 this is mode = BORDER_REFLECT_101.

    Upsampling should produce samples at even-numbered coordinates with
    coordinates starting at 0.

    Input:
        image -- height x width x channels image of type float32.
    Output:
        up -- 2*height x 2*width x channels image of type float32.
    """
    # Saving original image height and width info.
    original_height = image.shape[0]
    original_width = image.shape[1]
    # Upsampling the image.
    # Creating sequence from 1 to N, where N is the number of columns in the image.
    obj = np.arange(1, image.shape[1] + 1, dtype=int)
    # Adding a zero between every element in each row of pixels.
    #
    # Note: Thanks to Andreas K. for his elegant approach to inserting zeros between
    # elements in a numpy array. See his post here: https://stackoverflow.com/a/53179919.   
    image = np.insert(arr=image, obj=obj, values=0, axis=1)
    # Creating an image of all zeros with the desired shape of the image after upsampling.
    upsampled_image = np.zeros((original_height * 2, original_width * 2, image.shape[2]))
    # Filling alternate rows (starting from row at index 0) with rows from the image.
    upsampled_image[::2] = image

    # Creating 1D convolution kernel.
    kernel = np.array([[1, 4, 6, 4, 1]]) / 8
    # Array that will hold the result of the pre-filtering using the gaussian kernel.
    # Each element in the array will be the result for a particular color channel, so in the case of an RGB
    # image, this array will hold three arrays of shape (height, width), where the first array is the result
    # for R-channel, second array is the result for G-channel, and third array is result for B-channel.
    convolve_result = []
    # Iterating over the indices of all the channels in the image.
    for channel_index in range(upsampled_image.shape[2]):
        # Grabbing the pixels corresponding only to the channel we are currently processing.
        sub_image = upsampled_image[:, :, channel_index]
        # Applying 1D convolution kernel along x direction.
        sub_image = scipy.ndimage.correlate(sub_image, kernel, mode='mirror')
        # Applying 1D convolution kernel along y direction.
        sub_image = scipy.ndimage.correlate(sub_image, kernel.T, mode='mirror')
        # Adding result to 'convolve_result'.
        convolve_result.append(sub_image)
    # Combining different channels back together into one single image.
    result = np.dstack(convolve_result)
    return result

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # Getting projection matrix P.
    P = np.matmul(K, Rt)
    # Saving the height and width of the original 'points' array.
    points_height = points.shape[0]
    points_width = points.shape[1]
    # Reshaping 'points' to be of shape (height * width, 3).
    points = points.reshape(points_height*points_width, 3)
    # Adding 1 to the end of every row (to turn 3D points to homogeneous coordinates).
    points = np.insert(points, 3, 1, axis=1)
    # Performing matrix multiplication to project all the homogeneous coordinates.
    projections = np.matmul(P, points.T)
    # Changing orientation of 'projections'.
    projections = projections.T
    def convert_to_img_coords(row):
        """
        Each 'row' is a projected homogeneous coordinate (as a result of our matrix multiply).
        This function converts the homogeneous coordinate into its respective 2D image coordinate
        and returns the result as a 2D array.  
        """
        x = row[0] / row[2]
        y = row[1] / row[2]
        return [x, y]
    # Applying the function we defined above to every row of 'projections'.
    projections = np.apply_along_axis(convert_to_img_coords, 1, projections)
    # Reshaping 'projections' to the desired output shape. 
    projections = projections.reshape(points_height, points_width, 2)
    return projections


def unproject_corners_impl(K, width, height, depth, Rt):
    """
    Undo camera projection given a calibrated camera and the depth for each
    corner of an image.

    The output points array is a 2x2x3 array arranged for these image
    coordinates in this order:

     (0, 0)      |  (width, 0)
    -------------+------------------
     (0, height) |  (width, height)

    Each of these contains the 3 vector for the corner's corresponding
    point in 3D.

    Tutorial:
      Say you would like to unproject the pixel at coordinate (x, y)
      onto a plane at depth z with camera intrinsics K and camera
      extrinsics Rt.

      (1) Convert the coordinates from homogeneous image space pixel
          coordinates (2D) to a local camera direction (3D):
          (x', y', 1) = K^-1 * (x, y, 1)
      (2) This vector can also be interpreted as a point with depth 1 from
          the camera center.  Multiply it by z to get the point at depth z
          from the camera center.
          (z * x', z * y', z) = z * (x', y', 1)
      (3) Use the inverse of the extrinsics matrix, Rt, to move this point
          from the local camera coordinate system to a world space
          coordinate.
          Note:
            | R t |^-1 = | R^T -R^T t |
            | 0 1 |      | 0      1   |

          p = R^T * (z * x', z * y', z) - R^T t

    Input:
        K -- camera intrinsics calibration matrix
        width -- camera width
        height -- camera height
        depth -- depth of plane with respect to camera
        Rt -- 3 x 4 camera extrinsics calibration matrix
    Output:
        points -- 2 x 2 x 3 array of 3D points
    """
    # Array of the corners of the image. Each element is the 2D image coordinate
    # of one of the corners of the image. The order of the array is...
    # left-top corner, right-top corner, left-bottom corner, right-bottom corner.
    corners = [[0, 0], [width, 0], [0, height], [width, height]]
    # Each element of this array will be a 3D array that represents a corner's
    # corresponding point in 3D. Order of the corresponding points will follow the same order
    # as the order of the corners in 'corners' defined above. 
    output_points = []

    # Iterating over all the corners, undoing camera projection, and adding the 3D real world
    # point to 'output_points'.
    for corner in corners:
        # Turning 2D-image coordinate into homogeneous coordinate.
        corner.append(1)
        # Converting the coordinates from homogeneous image space pixel
        # coordinates (2D) to a local camera direction (3D).
        local_camera_direction_coord = np.matmul(np.linalg.inv(K), corner)
        # Multiply 'local_camera_direction_coord' by z to get the point at depth z
        # from the camera center.
        depth_z_coord = local_camera_direction_coord * depth
        # Using the inverse of the extrinsics matrix, Rt, to move this point
        # from the local camera coordinate system to a world space coordinate.
        R_transpose = Rt[:, :3].T
        t = Rt[:, 3]
        p =  np.matmul(R_transpose, depth_z_coord) - np.matmul(R_transpose, t)
        # Adding 'p' to 'output_points'.
        output_points.append(p)

    # Converting 'output_points' to a numpy array. 
    output_points = np.array(output_points)
    # Reshaping 'output_points' to desired return shape.
    output_points = output_points.reshape(2, 2, 3)
    return output_points

def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region.
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    # List containing the normalized vector for each pixel in the image. Each element is a normalized vector
    # corresponding to a particular pixel in the image. The order of elements will be row-major order of the pixels
    # in the image.
    normalized_vectors = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Initializing patch of zeros.
            patch = np.zeros(image.shape[2] * ncc_size**2)
            # Checking if the patch at the current pixel (we are processing) will be within the boundary of 
            # the input image.
            delta = int((ncc_size - 1) / 2)
            if i - delta >= 0 and j - delta >= 0 and i + delta + 1 <= image.shape[0] and j + delta + 1 <= image.shape[1]:
                # Extracting patch in image corresponding to current pixel we are processing.
                #
                # Note: pixel is located at the center of the patch.
                patch = image[i - delta:i + delta + 1, j - delta:j + delta + 1]
                # Reshaping patch so that 1D array of pixels (rather than a 2D grid).
                patch = patch.reshape(ncc_size**2, image.shape[2])
                # Calculating the mean of the intensity values for each color channel.
                channel_means = np.mean(patch, axis=0)
                # For each channel, subtracting the channel mean from each value belonging to that channel.
                patch = patch - channel_means
                # Tranposing 'patch' so vector elements will ultimately be in the right order (such that first channel
                # values are first, then second, and so on).
                patch = patch.T
                # Flattening patch to 1D array.
                patch = patch.flatten()
                # Computing the norm over the entire patch.
                patch_norm = np.linalg.norm(patch)
                # If norm is less than 1e6, setting 'patch' to zero vector, otherwise dividing 'patch'
                # by its norm.
                if patch_norm < 1e-6:
                    patch = np.zeros(image.shape[2] * ncc_size**2)
                else:
                    patch = patch / patch_norm
            # Adding 'patch' to 'normalized_vectors'.
            normalized_vectors.append(patch)
    # Converting 'normalized_vectors' to a numpy array.
    normalized_vectors = np.array(normalized_vectors)
    # Reshaping 'normalized_vectors' to desired output shape.
    normalized_vectors = normalized_vectors.reshape(image.shape[0], image.shape[1], image.shape[2] * ncc_size**2)
    return normalized_vectors

def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # Extracting and saving the length of each individual dimension of the image.
    height = image1.shape[0]
    width = image1.shape[1]
    patch_vector_length = image1.shape[2]
    # Reshaping both images to be a 1D array of patch vectors (as opposed to a 2D-grid).
    image1 = image1.reshape(height * width, patch_vector_length)
    image2 = image2.reshape(height * width, patch_vector_length)
    # For each matching pixel location in both images, computing the dot product between the respective patch
    # vectors for that location. 
    output = [np.dot(image1[i], image2[i]) for i in range(height * width)]
    # Converting 'output' to numpy array.
    output = np.array(output)
    # Reshaping 'output' to desired output shape.
    output = output.reshape(height, width)
    return output

