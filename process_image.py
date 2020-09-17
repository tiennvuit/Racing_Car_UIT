import cv2
import numpy as np


def make_points(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # left_fit_average = np.average(left_fit, axis = 0)
    # right_fit_average = np.average(right_fit, axis = 0)

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        print(left_fit_average, 'left')
        left_line = make_points(image, left_fit_average)
    
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        print(right_fit_average, 'right')
        right_line = make_points(image, right_fit_average)

    return np.array((left_line, right_line))


def CannyEdge(image):
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  cannyImage = cv2.Canny(blur, 50, 150)
  return cannyImage

def region_of_interest(image):
	height = image.shape[0]
	triangle = np.array([[(0, height-10),(160, 90),(275, height-10),]], np.int32)
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, triangle, 255)
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


