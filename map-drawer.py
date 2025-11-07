import math
import numpy as np
import cv2

rslt_img_x = 250
rslt_img_y = 250

line_range = 200
line_width = 200

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

def rad2degree(angle):
    return angle / math.pi * 180

def degree2rad(angle):
    return angle / 180 * math.pi

# draw utils
def draw_lidar(theta, d):
    img = np.zeros((rslt_img_x, rslt_img_y, 3), dtype=np.uint8)
    center_x = rslt_img_x // 2
    center_y = rslt_img_y // 2

    for i in range(len(theta)):
        x, y = polar_to_cartesian(d[i], degree2rad(theta[i]))

        px = int(center_x + x)
        py = int(center_y - y)  # Invert y-axis for image coordinates
        if 0 <= px < rslt_img_x and 0 <= py < rslt_img_y:
            img = cv2.circle(img, (px, py), 1, lidar_color, -1)
    
    return img

def draw_canny(img):
    # img : numpy array - cannyed image
    w = img.shape[1]
    h = img.shape[0]

    src_points = np.float32([
        [180, 280],  # 좌측 상단
        [460, 280],  # 우측 상단
        [800, 400], # 우측 하단
        [-160, 400]  # 좌측 하단
    ]) # 수정 필요

    dst_w = line_width
    dst_h = int(rslt_img_y // 2 - line_range)  

    dst_points = np.float32([
        [0, 0],
        [dst_w, 0],
        [dst_w, dst_h],
        [0, dst_h]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_img = cv2.warpPerspective(img, matrix, (dst_w, dst_h))

    # change color to line_color
    warped_img_colored = cv2.cvtColor(warped_img, cv2.COLOR_GRAY2BGR)
    warped_img_colored[np.where((warped_img_colored != [0, 0, 0]).all(axis=2))] = line_color

    return warped_img

def draw_map(img, theta, d):
    lidar_img = draw_lidar(theta, d)
    cannyed_img = draw_canny(img)

    roi = lidar_img[0:cannyed_img.shape[0], rslt_img_x//2 - cannyed_img.shape[1]//2:rslt_img_x//2 + cannyed_img.shape[1]//2]
    roi[np.where((cannyed_img != [0, 0, 0]).all(axis=2))] = cannyed_img[np.where((cannyed_img != [0, 0, 0]).all(axis=2))]

    lidar_img[0:cannyed_img.shape[0], rslt_img_x//2 - cannyed_img.shape[1]//2:rslt_img_x//2 + cannyed_img.shape[1]//2] = roi
    rslt_img = lidar_img
    
    return rslt_img