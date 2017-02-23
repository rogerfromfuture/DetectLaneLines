# importing some useful packages
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_one_line(img, lines, color=[255, 0, 0], thickness=10):
    image_shape = img.shape
    bottom_left_x = image_shape[1] + 1
    bottom_left_y = 0
    bottom_right_x = 0
    bottom_right_y = 0
    left_slope_array = []
    right_slope_array = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                bottom_left_x = min(bottom_left_x, x1, x2)
                bottom_left_y = max(bottom_left_y, y1, y2)
                left_slope_array.append(slope)
            elif slope > 0:
                bottom_right_x = max(bottom_right_x, x1, x2)
                bottom_right_y = max(bottom_right_y, y1, y2)
                right_slope_array.append(slope)

    upper_y_boundary = 350
    if len(left_slope_array) > 0:
        leftPercentileSlope = np.percentile(left_slope_array, 50)
        topLeftX = int((upper_y_boundary - bottom_left_y) / leftPercentileSlope + bottom_left_x)
        topLeftY = upper_y_boundary
        cv2.line(img, (bottom_left_x, bottom_left_y), (topLeftX, topLeftY), color, thickness)

    if len(right_slope_array) > 0:
        rightPercentileSlope = np.percentile(right_slope_array, 50)
        topRightX = int((upper_y_boundary - bottom_right_y) / rightPercentileSlope + bottom_right_x)
        topRightY = upper_y_boundary
        cv2.line(img, (bottom_right_x, bottom_right_y), (topRightX, topRightY), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, thickness=2, raw_hough_lines=True):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if raw_hough_lines:
        draw_lines(line_img, lines, [255, 0, 0], thickness)
    else:
        draw_one_line(line_img, lines, [255, 0, 0], thickness)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def process_image(image, raw_hough_lines=False, thickness=10):
    # Grayscale original image
    processed_image = grayscale(image)

    # Apply gaussian blurring
    # Define a kernel size for Gaussian smoothing / blurring
    kernel_size = 5
    processed_image = gaussian_blur(processed_image, kernel_size)

    # Apply Canny
    # Define parameters for Canny and run it
    # NOTE: if you try running this code you might want to change these!
    low_threshold = 50
    high_threshold = 150
    processed_image = canny(processed_image, low_threshold, high_threshold)

    # Define region
    imshape = image.shape
    vertices = np.array([[(120, imshape[0]), (470, 317), (470, 317), (imshape[1], imshape[0])]], dtype=np.int32)
    processed_image = region_of_interest(processed_image, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 30  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    processed_image = hough_lines(processed_image, rho, theta, threshold, min_line_length, max_line_gap, thickness, raw_hough_lines)
    processed_image = weighted_img(processed_image, image)

    return processed_image

def save_processed_images():
    fileNames = os.listdir("test_images/")

    for fileName in fileNames:
        fileOriginalFullPath = 'test_images/' + fileName
        image = mpimg.imread(fileOriginalFullPath)

        # Print out original image
        print('This image is:', type(image), 'with dimesions:', image.shape)
        plt.imshow(image)

        # Print out processed image
        fileProcessedFullPath = 'test_images/processed_' + fileName
        processed_image = process_image(image, True, 2)
        plt.imshow(processed_image)

        if os.path.exists(fileProcessedFullPath):
            os.remove(fileProcessedFullPath)

        # Save processed image to test_images/
        mpimg.imsave(fileProcessedFullPath, processed_image)


def show_processed_image(image):
    processed_image = process_image(image)
    plt.imshow(processed_image)


def save_videos():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip

    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip('solidYellowLeft.mp4')
    yellow_clip = clip2.fl_image(process_image)
    yellow_clip.write_videofile(yellow_output, audio=False)


plt.interactive(True)
image = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
print('This image is:', type(image), 'with dimesions:', image.shape)
show_processed_image(image)
#save_processed_images()
save_videos()
pass
