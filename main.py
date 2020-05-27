"""
Advanced Lane Detection Term 1
Self Driving Car NanoDegree

"""
# Pipeline for Advance Lane Detection

# Importing Dependencies
import sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#---------------------------------------

# Importing Modules
from perspective_transform import PerspectiveTransform
from line import Line
from line_detector import LineDetector

# this function computes camera Calibration Parameters
# 1. Calibration Matrix
# 2. Distortion Coefficients
def GetCalibrationParam(image_url):

    images = glob.glob(image_url)   #store images

    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    object_points = [] # 3d Points in real world space
    image_points = [] # 2d Points in image plane.
    corner = (9, 6) # Chessboard size to 9x6

    # Iterate over the stored images
    for image in images:
        img = mpimg.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, corner, None)

        if ret:
            object_points.append(objp)
            image_points.append(corners)

    img_size = (img.shape[1], img.shape[0])

    # Here, we will use built in cv2 function named as calibrateCamera
    # This function finds the camera intrinsic and extrinsic parameters...
    # from several views of a calibration pattern
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size,None,None)

    # Return Calibration matrix and Distortion Matrix
    return mtx, dist

# This function gets camera callibration matrix and distortion coefficents from the saved configuration file.
def InputCalibrationFile(path="./cal.npz"):

    try:
        calibration_param = np.load(path)
        return calibration_param['mtx'], calibration_param['dist']
    except IOError as e:
        print(e)    # Throw exception
        raise IOError("Please Set Correct Calibration File")

# This function Calculates Direct Threshold
def DirectThreshold(img_ch, sobel_kernel=3, thresh=(0, np.pi/2)):

    sobelx = np.absolute(cv2.Sobel(img_ch, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    sobely = np.absolute(cv2.Sobel(img_ch, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    abs_grad_dir = np.absolute(np.arctan(sobely/sobelx))
    dir_binary =  np.zeros_like(abs_grad_dir)
    dir_binary[(abs_grad_dir > thresh[0]) & (abs_grad_dir < thresh[1])] = 1

    return dir_binary

def main():
    camera_matrix, dist_coeff = GetCalibrationParam('./camera_cal/calibration*.jpg')
    np.savez("./cal.npz",mtx=camera_matrix, dist=dist_coeff)
    output = './output_video.mp4'
    clip1 = VideoFileClip('./project_video.mp4')
    # Initiate the LineDetector Object
    ld = LineDetector()
    # Process Images
    white_clip = clip1.fl_image(ld.ProcessImage)
    white_clip.write_videofile(output, audio=False)
    print("Success, process Finished. Please see %s" % output)

if __name__ == "__main__":
    main()