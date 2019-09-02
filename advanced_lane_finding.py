# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
from IPython import get_ipython

#%% [markdown]
# ## Advanced Lane Finding Project
# 
# The goals / steps of this project are the following:
# 
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# ---
# ## Calibrate the Camera

#%%
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')
# Helper Methods
def save_image(fname, img):
    cv2.imwrite('./output_images/{}.jpg'.format(fname), img)
    
def display_images(images, cols = 4, rows = 5, figsize=(15,10), cmap = None):
    """
    Show a list of images in grid format using matplotlib
    """
    img_length = len(images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    indexes = range(cols * rows)

    for ax, index in zip(axes.flat, indexes):
        if index < img_length:
            image_path, image = images[index]
            if cmap == None:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap=cmap)
                ax.set_title(image_path)
                ax.axis('off')
            
def display_comparison(img_l, title_l, img_r, title_r, figsize=(20,10) , cmap_l=None, cmap_r=None):
    """
    Display a comparison of the output of some transform on a specified image.
    """
    fig, axes = plt.subplots(ncols=2, figsize=figsize)
    left_axes, right_axes = axes
    
    if cmap_l == None:
        left_axes.imshow(img_l)
    else:
        left_axes.imshow(img_l, cmap=cmap_l)
        left_axes.set_title(title_l)
        left_axes.axis('off')
    if cmap_r == None:
        right_axes.imshow(img_r)
    else:
        right_axes.imshow(img_r, cmap=cmap_r)
        right_axes.set_title(title_r)
        right_axes.axis('off')
    
def convert_to_rgb(img):
    return cv2.Color(img, cv2.COLOR_BGR2RGB)


#%%
# Load images from test images folder and store for later use.
calibration_images = list(map(
    lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
    glob.glob('./camera_cal/*.jpg')
))

display_images(calibration_images, 4, 5, (15, 13))


#%%
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

originals = []
drawn_corners = []

# Step through the list and search for chessboard corners
for fname, img in calibration_images:
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img_copy = cv2.drawChessboardCorners(img_copy, (9,6), corners, ret)
        
        originals.append(img)
        drawn_corners.append(img_copy)
        
index = 2
display_comparison(
    originals[index],
    'Original',
    drawn_corners[index],
    'With Corners'
)


#%%
img_shape = originals[index].shape[0:2]

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
undist = cv2.undistort(originals[index], mtx, dist, None, mtx)

display_comparison(
    originals[index],
    'Original',
    undist,
    'Undistorted'
)


#%%
# Loading camera calibration
cameraCalibration = pickle.load( open('./calibration.p', 'rb' ) )
mtx, dist = map(cameraCalibration.get, ('mtx', 'dist'))


#%%
# Load images from test images folder and store for later use.
test_images = list(map(
    lambda imageFileName: (imageFileName, cv2.imread(imageFileName)),
    glob.glob('./test_images/*.jpg')
))
for path, img in test_images:
    print(path)

#%% [markdown]
# ## Undistort Images with Calibrated Camera

#%%
display_images(list(map(lambda img: (img[0], cv2.cvtColor(img[1], cv2.COLOR_BGR2RGB)), test_images)), 2, 3, (15, 13))


#%%
original = test_images[-1][1]
display_comparison(
    cv2.cvtColor(original, cv2.COLOR_BGR2RGB), 'Original',
    cv2.cvtColor(cv2.undistort(original, mtx, dist, None, mtx), cv2.COLOR_BGR2RGB), 'Undistort'
)


#%%
def convert_to_hls(image, mtx=mtx, dist=dist):
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    return cv2.cvtColor(undist, cv2.COLOR_BGR2HLS)


#%%
get_s_channel = lambda img: convert_to_hls(img)[:,:,2]

#%% [markdown]
# ## Transform Perspective to Find Lane Curvature

#%%
original = cv2.cvtColor(test_images[-1][1],cv2.COLOR_BGR2RGB)
undist = cv2.undistort(original, mtx, dist, None, mtx)
copy = undist.copy()

def draw_line(img, x,  y): 
    cv2.line(img, x, y, [255, 0, 0], 2)
    
def display_large_image(img, cmap=None):
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.imshow(img, cmap=cmap)

bottom_y = 720
top_y = 455

L1 = (200, bottom_y)
L2 = (590, top_y)
R1 = (695, top_y)
R2 = (1120, bottom_y)

L1_x, L1_y = L1
L2_x, L2_y = L2
R1_x, R1_y = R1
R2_x, R2_y = R2

draw_line(copy, L1, L2)
draw_line(copy, L2, R1)
draw_line(copy, R1, R2)
draw_line(copy, R2, L1)

display_large_image(copy)


#%%
gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

ny, nx = gray.shape
img_size = (nx, ny)
offset = 350

src = np.float32([ 
    [L2_x, L2_y],
    [R1_x, R1_y],
    [R2_x, R2_y],
    [L1_x, L1_y]
])

dst = np.float32([
    [offset, 0],
    [nx - offset, 0],
    [nx - offset, ny], 
    [offset, ny]
])

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_NEAREST)

draw_line(warped, (offset, 0), (nx-offset, 0))
draw_line(warped, (nx-offset, 0), (nx-offset, ny))
draw_line(warped, (nx-offset, ny), (offset, ny))
draw_line(warped, (offset, ny), (offset, 0))

display_comparison(
    copy, 'Original',
    warped, 'Perspective transformed'
)


#%%
fig, axes = plt.subplots(ncols=3, figsize=(20,10))
hlsImg = convert_to_hls(original)
for index, a in enumerate(axes):
    a.imshow(hlsImg[:,:,index], cmap='gray')
    a.axis('off')


#%%
def transform_collection_images(collection, transform):
    return list(map(lambda img_data: (img_data[0], transform(img_data[1])), collection))

def display_transformed_images(collection, transform):
    result = transform_collection_images(collection, transform)
    display_images(result, 2, 3, (15, 13), cmap='gray')
    return result

s_channel_images = display_transformed_images(test_images, get_s_channel)

#%% [markdown]
# # Apply Binary Thresholds to Determine Lane Lines
#%% [markdown]
# # Sobel X

#%%
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    
    thresh_min, thresh_max = thresh
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return binary_output

sobel_config = {'sobel_kernel': 15, 'thresh': (10, 200)}

apply_sobel_x = lambda image: abs_sobel_thresh(get_s_channel(image), orient='x', **sobel_config)
sobel_x = display_transformed_images(test_images, apply_sobel_x)

#%% [markdown]
# ## Sobel Y

#%%
apply_sobel_y = lambda image: abs_sobel_thresh(get_s_channel(image), orient='y', **sobel_config)
sobel_y = display_transformed_images(test_images, apply_sobel_y)

#%% [markdown]
# ## Gradient Magnitude

#%%
def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return mag_binary

mag_config = {'sobel_kernel': 15, 'thresh': (20, 100)}

apply_magnitude = lambda image: mag_threshold(get_s_channel(image), **mag_config)
magnitude = display_transformed_images(test_images, apply_magnitude)

#%% [markdown]
# ## Direction Threshold

#%%
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return dir_binary

dir_config = {'sobel_kernel': 15, 'thresh': (0.7, 1.20)}

apply_direction = lambda image: dir_threshold(get_s_channel(image), **dir_config)
direction = display_transformed_images(test_images, apply_direction)


#%%
def apply_binary_thresholding(img):
    sobel_x = apply_sobel_x(img)
    sobel_y = apply_sobel_y(img)
    mag = apply_magnitude(img)
    direction = apply_direction(img)
    combined = np.zeros_like(sobel_x) 
    combined[((sobel_x == 1) & (sobel_y == 1) | (mag == 1) & (direction == 1))] = 1
 
    return combined
    
combined_thresholding = display_transformed_images(test_images, apply_binary_thresholding)


#%%
titles = ['SobelX', 'SobelY', 'Magnitude', 'Direction', 'Combined']
thresholds = list(zip( sobel_x, sobel_y, magnitude, direction, combined_thresholding ))

titled_thresholds = list(map(lambda images: list(zip(titles, images)), thresholds))[3:6]
formatted_threshold_examples = [item for sublist in titled_thresholds for item in sublist]

fig, axes = plt.subplots(ncols=5, nrows=len(titled_thresholds), figsize=(45,20))
for ax, thresholds in zip(axes.flat, formatted_threshold_examples):
    title, images = thresholds
    image_path, img = images
    ax.imshow(img, cmap='gray')
    ax.set_title(image_path + '\n' + title, fontsize=12)
    ax.axis('off')
fig.subplots_adjust(hspace=0, wspace=0.05, bottom=0)

#%% [markdown]
# ## Detect Start of Lane Lines with Histogram

#%%
def warp(img, M=M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
    return warped

threshold_and_warp = lambda img: warp(apply_binary_thresholding(img))
transformed = display_transformed_images(test_images, threshold_and_warp)


#%%
# Create histogram of image binary activations
def hist(img):
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0]//2:,:]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram

histogram = hist(transformed[0][1])
plt.plot(histogram)

#%% [markdown]
# ## Detect Lane Lines Using a Sliding Window

#%%

class Lines():
    
    def __init__(self):
        self.left_fit = None
        self.right_fit = None
        self.left_fitx = None
        self.right_fitx = None
        self.ploty = None
        self.detected = False
        self.nonzeroy = None
        self.nonzerox = None
         # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        self.left_curv = None
        self.right_curv = None
        self.center_offset = None

        
    def find_lane(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),
            (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),
            (win_xright_high,win_y_high),(0,255,0), 2) 

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            self.left_lane_inds = np.concatenate(self.left_lane_inds)
            self.right_lane_inds = np.concatenate(self.right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds] 
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]
        
        self.detected = True

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img_shape, leftx, lefty, rightx, righty):
         # Fit a second order polynomial to each with np.polyfit()
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
        
        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        self.right_fit = right_fit
        self.left_fit = left_fit
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty
        
    def find_initial_lanes(self, binary_warped):
        leftx, lefty, rightx, righty, out_img = self.find_lane(binary_warped)
        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    def search_around_poly(self, binary_warped):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function
        # Hint: consider the window areas for the similarly named variables
        # in the previous quiz, but change the windows to our new search area
            
        self.left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        self.right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds] 
        rightx = nonzerox[self.right_lane_inds]
        righty = nonzeroy[self.right_lane_inds]

        # Fit new polynomials
        self.nonzeroy = nonzeroy
        self.nonzerox = nonzerox
        self.fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])

        return left_curverad, right_curverad


    def visualize_image(self, binary_warped):
        margin = 100
        ## Visualization ##
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([self.left_fitx-margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+margin, 
                                  self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([self.right_fitx-margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+margin, 
                                  self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # Plot the polynomial lines onto the image
        plt.plot(self.left_fitx, self.ploty, color='yellow')
        plt.plot(self.right_fitx, self.ploty, color='yellow')
        ## End visualization steps ##
        
        return result
    
    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*self.left_fit[0]*y_eval*ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        right_curverad = ((1 + (2*self.right_fit[0]*y_eval*ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        center = (((self.left_fit[0]*720**2+self.left_fit[1]*720+self.left_fit[2]) +(self.right_fit[0]*720**2+self.right_fit[1]*720+self.right_fit[2])) /2 - 640)*xm_per_pix

        self.left_curv = left_curverad
        self.right_curv = right_curverad
        self.center_offset = center
        
    def draw_curvature_info(self, img):
        self.measure_curvature_real()
        
        # Display lane curvature and center offset
        curvature_info = [
            "Left curv: {:.2f}m".format(self.left_curv),
            "Right curv: {:.2f}m".format(self.right_curv),
            "Center offset: {:.2f}m".format(self.center_offset)
        ]

        scale = 1.25
        thickness = 2
        font = cv2.FONT_HERSHEY_DUPLEX
        color = (255,255,255)
        lineType = cv2.LINE_AA

        base = 40
        multiplier = 60

        for i, info in enumerate(curvature_info):
            line_height = base + (multiplier * i)
            cv2.putText(img, info, (130, line_height), font, scale, color, lineType=lineType)

        return img
    
    def find_lanes_on_road(self, img):
        binary = threshold_and_warp(img)
        if not self.detected:
            self.find_lane(binary)
        else:
            self.search_around_poly(binary)
    
    def draw_lanes_on_road(self, img):
        color_warp = np.zeros_like(img).astype(np.uint8)

        self.find_lanes_on_road(img)
        
        pts_left = np.array([np.transpose(np.vstack([self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))    
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        lanes = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

        return cv2.cvtColor(self.draw_curvature_info(lanes), cv2.COLOR_BGR2RGB)

test_img = transformed[3][1]
laneFinder = Lines()
laneFinder.find_initial_lanes(test_img)
laneFinder.search_around_poly(test_img)
output = laneFinder.visualize_image(test_img)

# plt.imshow(cv2.imread('./output_images/lanes_from_previous.jpg'))
# plt.savefig('./output_images/sliding_windows_with_plot.jpg',  bbox_inches='tight')
# display_large_image(cv2.imread('./output_images/sliding_windows.jpg'))
full_pipe = laneFinder.draw_lanes_on_road(test_images[0][1])
save_image('full_lane_pipe', full_pipe)

display_large_image(cv2.imread('./output_images/lanes_plot.jpg'))
display_large_image(cv2.imread('./output_images/sliding_windows_plot.jpg'))
display_large_image(cv2.imread('./output_images/full_lane_pipe.jpg'))


#%%
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  


#%%
from moviepy.editor import VideoFileClip
from IPython.display import HTML

f_name = 'project_video.mp4'
video = VideoFileClip('{}'.format(f_name))  # Load the original video
video = video.fl_image(laneFinder.draw_lanes_on_road)  # Pipe the video frames through the lane-detection pipeline
# video = video.subclip(39, 43)  # Only process a portion
get_ipython().run_line_magic('time', "video.write_videofile('./output_videos/{}'.format(f_name), audio=False)")


#%%



#%%



