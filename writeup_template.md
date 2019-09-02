**Advanced Lane Finding Project**

The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # 'Image References'
[image1]: ./output_images/undistorted7.jpg 'Undistorted'
[image2]: ./test_images/test1.jpg 'Road Transformed'
[image3]: ./output_images/binary.jpg 'Binary Example'
[image4]: ./output_images/warped.jpg 'Warp Example'
[image5]: ./output_images/sliding_windows_plot.jpg 'Fit Visual'
[image6]: ./output_images/lane_on_road.jpg 'Output'
[video1]: ./project_video.mp4 'Video'

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./advanced_lane_finding.ipynb" (or in called `advanced_lane_finding.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
By acquiring the camera matrix and destination points return from the `cv2.calibrateCamera()` function in the previous step, and feeding those to the `cv2.undistort()` function we can remove the distortion caused by various lens aberrations and ultimately produce an image free of any distortion.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image thresholding steps in advanced_lane_finding.ipynb (or in `advanced_lane_finder.py`). Here's an example of my output for this step. (note: this is not actually from one of the test images)

![alt text][image3]

After converting the original image into the HLS colorspace I separated the S channel and applied a Sobel threshold to it to identify strong pixels for the the x and y direction separately. I then combined those thresholds with others using methods such as taking the direction of the gradient as well as the magintude. Overall this produced a nicely thresholded image on which I could start to detect where the lane lines fell in each image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in the file `advanced_lane_finding.ipynb` (output_images/warped.py) (or in `advanced_lane_finding.py`). The `warp()` function takes as inputs an image (`img`). The (`src`) and (`dst`) were found and are accessible as global variables.

```python
bottom_y = 720
top_y = 455

L1 = (190, bottom_y)
L2 = (585, top_y)

L1_x, L1_y = L1
L2_x, L2_y = L2

R1 = (705, top_y)
R2 = (1130, bottom_y)

R1_x, R1_y = R1
R2_x, R2_y = R2

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)

ny, nx = gray.shape
img_size = (nx, ny)
offset = 200

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
```

This resulted in the following source and destination points:

|  Source   | Destination |
| :-------: | :---------: |
| 585, 455  |   200, 0    |
| 705, 455  |   1080, 0   |
| 1130, 720 |  1080, 720  |
| 190, 720  |  200, 720   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
