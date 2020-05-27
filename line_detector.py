import sys
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from line import Line
from perspective_transform import PerspectiveTransform

class LineDetector(object):
    # Constructor
    def __init__(self, calibration_path="./cal.npz"):
        self.left_line = Line()
        self.right_line = Line()
        src = np.array([[490, 482],[810, 482],
                        [1250, 720],[40, 720]], dtype=np.float32)
        dst = np.array([[0, 0], [1280, 0],
                        [1250, 720],[40, 720]], dtype=np.float32)
        self.bird_view_transformer =PerspectiveTransform(src, dst)
        self.yvals = np.arange(720)
        self.detected = None
        self.count = 0

        try:
            calibration_param = np.load(calibration_path)
            self.camera_mtx, self.dist_coeff = calibration_param['mtx'], calibration_param['dist']

        except IOError as e:
            print(e)    # Throw exception
            raise IOError("Please Set Correct Calibration File")
            sys.exit()

    def MaskImageByAverageLines(self, converted_img, width=30):
        image = np.zeros_like(converted_img)
        for yv, ll in zip(self.yvals, self.left_line.bestx):
            image[int(yv), int(ll-width):int(ll+width)] = converted_img[int(yv), int(ll-width):int(ll+width)]
        for yv, rl in zip(self.yvals, self.right_line.bestx):
            image[int(yv), int(rl-width):int(rl+width)] = converted_img[int(yv), int(rl-width):int(rl+width)]
        return image
    
    def GetFilteredImgAndCalPolynomial(self, converted_img, width=30):
        image = self.MaskImageByAverageLines(converted_img, width=width)
        _, self.left_line.current_fit = self.CalculatePolynomial(image, 0, int(image.shape[1] / 2))
        _, self.right_line.current_fit = self.CalculatePolynomial(image, int(image.shape[1] / 2), image.shape[1])
        return image

    def MaskImageByLines(self, original_image, width=10):
        image = np.zeros_like(original_image)
        for yv, ll in zip(self.yvals, self.left_line.bestx):
            image[int(yv), int(ll-width):int(ll+width)] = 1
        for yv, rl in zip(self.yvals, self.right_line.bestx):
            image[int(yv), int(rl-width) : int(rl+width)] = 1
        return image

    # Given an image, left_boundary, right_boundary, this function calculates and fits the polynomial on it
    def CalculatePolynomial(self, img, left_boundary, right_boundary):
        side_img = img[:, left_boundary: right_boundary].copy()
        index = np.where(side_img == 1)
        yvals = index[0]
        xvals = index[1] + left_boundary
        if xvals.size != 0:
            fit_equation = np.polyfit(yvals, xvals, 2)
            fit_line = fit_equation[0]*self.yvals**2 + fit_equation[1]*self.yvals + fit_equation[2]
            return fit_line, fit_equation
        else:
            return 0, np.array([10000., 100., 100.])

    def CheckParallel(self):
        return True if (np.abs(self.left_line.current_fit[0] - self.right_line.current_fit[0]) < 0.01) else False

    def CheckSimilarity(self, side='left'):
        if side == 'left':
            return True if (np.abs(self.left_line.current_fit[0] - self.left_line.best_fit[0]) < 0.0005) else False
        else:
            return True if (np.abs(self.right_line.current_fit[0] - self.right_line.best_fit[0]) < 0.0005) else False

    def CheckLine(self):
        return self.CheckSimilarity(side='right') and self.CheckSimilarity(side='left') and self.CheckParallel()

    def CalBestxAndFit(self, weight=0.2):
        self.left_line.best_fit = self.left_line.best_fit * (1 - weight) + self.left_line.current_fit * weight
        self.left_line.bestx = self.left_line.best_fit[0]*self.yvals**2 + self.left_line.best_fit[1]*self.yvals + self.left_line.best_fit[2]
        self.right_line.best_fit = self.right_line.best_fit * (1 - weight) + self.right_line.current_fit * weight
        self.right_line.bestx = self.right_line.best_fit[0]*self.yvals**2 + self.right_line.best_fit[1]*self.yvals + self.right_line.best_fit[2]
        self.count = 0
    
    # This function computes radius of curvature for each lane in meters
    def CalculateCurvature(self):
        
        # Conversion from pixels to meters
        # By Simply multiplying the number of pixels by 3.7/700 along x dim and 30/720 along y dim.
        ym_per_pix = 30/720
        xm_per_pix = 3.7/700

        left_fit_cr = np.polyfit(self.yvals * ym_per_pix, self.left_line.bestx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.yvals * ym_per_pix, self.right_line.bestx * xm_per_pix, 2)

        self.left_line.radius_of_curvature = ((1 + (2*left_fit_cr[0]*np.max(self.yvals) + left_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*left_fit_cr[0])
        self.right_line.radius_of_curvature = ((1 + (2*right_fit_cr[0]*np.max(self.yvals) + right_fit_cr[1])**2)**1.5) \
                                        /np.absolute(2*right_fit_cr[0])

    def AddCurvatureToImage(self, image):
        self.CalculateCurvature()
        curvature = int((self.left_line.radius_of_curvature+self.right_line.radius_of_curvature)/2)
        cv2.putText(
            image, 'Radius of Curvature {}(m)'.format(curvature),
                    (120,140), fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
        return image
    
    def AddPlaceToImage(self, image):
        place = (self.left_line.bestx[-1] + self.right_line.bestx[-1]) / 2
        diff_center = np.abs((image.shape[1] / 2 - place) * 3.7 / 700)

        if place > image.shape[1] / 2:
            cv2.putText(image, 'Vehicle is {:.2f}m left of center'.format(diff_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
        else:
            cv2.putText(image, 'Vehicle is {:.2f}m right of center'.format(diff_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 2)
        return image
    
    # We define a subfunction for calculating index of max sum value of each histogram.
    def GetMaxIndexHistogram(self, histogram, left_boundary, right_boundary, window_width=10):
        index_list = []
        side_histogram = histogram[left_boundary : right_boundary]
        for i in range(len(side_histogram) - window_width):
            index_list.append(np.sum(side_histogram[i : i + window_width]))
        index = np.argmax(index_list) + int(window_width / 2) + left_boundary
        return index

    # This function calculates Histogram Thresholding for decreasing noise from given binary images
    def HistogramThresholding(self, img, xsteps=20, ysteps=40, window_width=10):
    
        xstride = img.shape[0] // xsteps
        ystride = img.shape[1] // ysteps
        for xstep in range(xsteps):
            histogram = np.sum(img[xstride*xstep : xstride*(xstep+1), :], axis=0)
            boundary = int(img.shape[1] / 2)
            leftindex = self.GetMaxIndexHistogram(histogram, 0, boundary, window_width=window_width)
            rightindex = self.GetMaxIndexHistogram(histogram, boundary, img.shape[1], window_width=window_width)

            # mask the image
            if histogram[leftindex] >= 3:
                img[xstride*xstep : xstride*(xstep+1), : leftindex-ysteps] = 0
                img[xstride*xstep : xstride*(xstep+1), leftindex+ysteps+1 : boundary] = 0
            else:
                img[xstride*xstep : xstride*(xstep+1), : boundary] = 0

            if histogram[rightindex] >= 3:
                img[xstride*xstep : xstride*(xstep+1), boundary :rightindex-ysteps] = 0
                img[xstride*xstep : xstride*(xstep+1), rightindex+ysteps+1 :] = 0
            else:
                img[xstride*xstep : xstride*(xstep+1), boundary : ] = 0

        left_fit_line, left_line_equation = self.CalculatePolynomial(img, 0, boundary)
        right_fit_line, right_line_equation = self.CalculatePolynomial(img, boundary, img.shape[1])

        # Return binary image after histogram thresholding
        return img, left_fit_line, right_fit_line, left_line_equation, right_line_equation


    def SelectBestxAndFit(self, image):
        if self.detected:
            final_bird_view_img = self.GetFilteredImgAndCalPolynomial(image, width=50)
            if self.CheckLine():
                self.CalBestxAndFit()
            else:
                self.count += 1
                if self.count == 2:
                    self.detected = None
                    self.count = 0
        else:
            final_bird_view_img, left_line, right_line, left_line_equation, right_line_equation = \
                self.HistogramThresholding(image, xsteps=20, ysteps=25, window_width=15)
            self.left_line.current_fit = left_line_equation
            self.right_line.current_fit = right_line_equation

            if not self.CheckParallel():
                self.detected = False
            else:
                self.left_line.bestx = left_line
                self.left_line.best_fit = left_line_equation
                self.right_line.bestx = right_line
                self.right_line.best_fit = right_line_equation
                self.detected = True
        return final_bird_view_img

    # This function returns undistorted images using the calibration and distortion matrices
    def GetUndistortion(self, distorted_img, mtx, dist):
        # Here, we will use the built in cv2 function named as undistort
        # This function transforms an image to compensate for lens distortion.
        undist = cv2.undistort(distorted_img, mtx, dist, None, mtx)

        return undist   #return the compenstaed image

    # This function applies Gaussian Noise Kernal to the given binary image
    def GaussianBlurr(self, img, kernel_size):
        # We use cv2 function GaussianBlur
        # The Gaussian filter is a low-pass filter that removes the high-frequency components are reduced.
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Computes S Binary Image (x, y, 1) from H and S of HLS.
    def GetSBinary(self, undist_img, thres=(110, 255)):

        # Use cv2.cvtColor and RGV2HLS Parameter
        hls = cv2.cvtColor(undist_img, cv2.COLOR_RGB2HLS)
        h = hls[:, :, 0]
        s = hls[:, :, 2]
        s_binary = np.zeros_like(s)
        s_binary[((s >= thres[0]) & (s <= thres[1])) & (h <= 30)] = 1
        s_binary = self.GaussianBlurr(s_binary, kernel_size=21)

        return s_binary

    # This function returns RGB image after applying gamma conversion
    def AdjustGamma(self, image, gamma=1.0):

        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        
        # We wll use a look up table LUT from cv2
        return cv2.LUT(image, table)

    # This function is basically for Gradent Thresholding
    # We First apply gamma conversion to imags for decrease noise using 'AdjustGamma' function.
    def GetSlope(self, undist_img, orient='x', sobel_kernel=3, thres = (0, 255)):
    
        undist_img = self.AdjustGamma(undist_img, 0.2)
        gray = cv2.cvtColor(undist_img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            slope = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        elif orient == 'y':
            slope = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        else:
            raise KeyError("select 'x' or 'y'")

        scale_factor = np.max(slope) / 255
        scale_slope = (slope / scale_factor).astype(np.uint8)
        slope_binary = np.zeros_like(scale_slope)
        slope_binary[(scale_slope >= thres[0]) & (scale_slope <= thres[1])] = 1

        slope_binary = self.GaussianBlurr(slope_binary, kernel_size=9)
        return slope_binary

    # Given an undistorted image, return converted image using Color Slope Conversion
    def ColorSlopeThresConversion(self, undist_img):
        s_binary =self.GetSBinary(undist_img, thres=(150, 255))
        slope = self.GetSlope(undist_img, orient='x',sobel_kernel=7, thres=(25, 255))

        color_binary = np.zeros_like(s_binary)
        color_binary[(s_binary == 1) | (slope == 1)] = 1

        return color_binary
    
    # This function returns adds lines to the given new image
    # Here, we use cv2 function called 'fillConvexPoly'.
    def AddLinesToImage(self, undist_img, new_image):
        index = np.where(new_image == 1)
        pt = np.vstack((index[1], index[0]))
        pt = np.transpose(pt)
        cv2.fillConvexPoly(undist_img, pt, (255, 93, 74))
        undist_img[new_image==1] = [255, 0, 0]
        return undist_img
    
    # Process the Image and return the undistored image
    def ProcessImage(self, distorted_image):
        undist_img = self.GetUndistortion(distorted_image, self.camera_mtx, self.dist_coeff)
        bird_view = self.bird_view_transformer.Transform(undist_img)
        converted_img = self.ColorSlopeThresConversion(bird_view)
        final_bird_view_img = self.SelectBestxAndFit(converted_img)
        final_bird_view_img = self.MaskImageByLines(final_bird_view_img)
        new_image = self.bird_view_transformer.InverseTransform(final_bird_view_img)
        undist_img = self.AddLinesToImage(undist_img, new_image)
        undist_img = self.AddCurvatureToImage(undist_img)
        undist_img = self.AddPlaceToImage(undist_img)
        return undist_img