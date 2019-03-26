import numpy as np
import cv2
import matplotlib.pyplot as plt


def create_feature(rgb_image):
  '''Basic brightness feature, required by Udacity'''
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) # Convert to HSV color space

  sum_brightness = np.sum(hsv[:,:,2]) # Sum the brightness values
  area = 32*32
  avg_brightness = sum_brightness / area # Find the average

  return avg_brightness

def high_saturation_pixels(rgb_image, threshold):
  '''Returns average red and green content from high saturation pixels'''
  high_sat_pixels = []
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  for i in range(32):
    for j in range(32):
      if hsv[i][j][1] > threshold:
        high_sat_pixels.append(rgb_image[i][j])

  if not high_sat_pixels:
    return highest_sat_pixel(rgb_image)

  sum_red = 0
  sum_green = 0
  for pixel in high_sat_pixels:
    sum_red += pixel[0]
    sum_green += pixel[1]

  # TODO: Use sum() instead of manually adding them up
  avg_red = sum_red / len(high_sat_pixels)
  avg_green = sum_green / len(high_sat_pixels) * 0.8 # 0.8 to favor red's chances
  return avg_red, avg_green

def highest_sat_pixel(rgb_image):
  '''Finds the higest saturation pixel, and checks if it has a higher green
  content, or a higher red content'''

  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  s = hsv[:,:,1]

  x, y = (np.unravel_index(np.argmax(s), s.shape))
  if rgb_image[x, y, 0] > rgb_image[x,y, 1] * 0.9: # 0.9 to favor red's chances
    return 1, 0 # Red has a higher content
  return 0, 1

def estimate_label(rgb_image): # Standardized RGB image
  return red_green_yellow(rgb_image)

def findNonZero(rgb_image):
  rows, cols, _ = rgb_image.shape
  counter = 0

  for row in range(rows):
    for col in range(cols):
      pixel = rgb_image[row, col]
      if sum(pixel) != 0:
        counter = counter + 1

  return counter

def red_green_yellow(rgb_image):
  '''Determines the Red, Green, and Yellow content in each image using HSV and
  experimentally determined thresholds. Returns a classification based on the
  values.
  '''
  hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
  sum_saturation = np.sum(hsv[:,:,1]) # Sum the brightness values
  area = 32*32
  avg_saturation = sum_saturation / area # Find the average

  sat_low = int(avg_saturation * 1.3)
  val_low = 140

  # Green
  lower_green = np.array([70,sat_low,val_low])
  upper_green = np.array([100,255,255])
  green_mask = cv2.inRange(hsv, lower_green, upper_green)
  green_result = cv2.bitwise_and(rgb_image, rgb_image, mask = green_mask)

  # Yellow
  lower_yellow = np.array([10,sat_low,val_low])
  upper_yellow = np.array([60,255,255])
  yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
  yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask = yellow_mask)

  # Red
  lower_red = np.array([150,sat_low,val_low])
  upper_red = np.array([180,255,255])
  red_mask = cv2.inRange(hsv, lower_red, upper_red)
  red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)

  sum_green = findNonZero(green_result)
  sum_yellow = findNonZero(yellow_result)
  sum_red = findNonZero(red_result)

  if sum_red >= sum_yellow and sum_red >= sum_green:
    # return [1,0,0] # Red
    return "red"
  if sum_yellow >= sum_green:
    # return [0,1,0] # Yellow
    return "yellow"
  # return [0,0,1] # Green
  return "green"

  # f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (20,10))
  # ax1.imshow(rgb_image)
  # ax2.imshow(red_result)
  # ax3.imshow(yellow_result)
  # ax4.imshow(green_result)
  # ax5.imshow(hsv)
  # plt.show()
