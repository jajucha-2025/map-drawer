import math
import numpy as np
import cv2

rslt_img_x = 250
rslt_img_y = 250

lidar_color = (0, 255, 0)
line_color = (255, 0, 0)

# simple math
def cartesian_to_polar(x, y):
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y, x)
    return r, theta

def polar_to_cartesian(r, theta):
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    return x, y

# draw utils
def draw_lidar(theta, d):
    img = np.zeros((rslt_img_x, rslt_img_y, 3), dtype=np.uint8)
    center_x = rslt_img_x // 2
    center_y = rslt_img_y // 2

    for i in range(len(theta)):
        x, y = polar_to_cartesian(d[i], theta[i])

        px = int(center_x + x[i])
        py = int(center_y - y[i])  # Invert y-axis for image coordinates
        if 0 <= px < rslt_img_x and 0 <= py < rslt_img_y:
            img = cv2.circle(img, (px, py), 1, lidar_color, -1)
    
    return img

def draw_canny(img):
    # img : numpy array - cannyed image
    filter = [
        [0, 1],
        [1, 0],
    ]