import math
import numpy as np
import cv2
import os

# -------------------------------
# ∴∵∴∵∴∵∴∵∴∵∴|↖∵∴∵∴∵∴∵∴∵∴∵∴∵∴∵∴
# ∴∵∴∵∴∵∴∵∴∵∴|∴∵↖∵∴∵∴∵∴∵∴∵∴∵∴∵∴
# ∴∵∴∵∴∵∴∵∴∵∴|∴∵∴∵↖∵∴∵∴∵∴∵∴∵∴∵∴
# ∴∵∴∵∴∵∴∵∴∵∴|400∵∴∵↖∵∴∵_∵∴┌-┐∴
# ∴∵∴∵∴∵∴∵∴∵∴|∴∵∴∵∴∵↙∵∴|└┐┌┘-└┐
# ∴∵∴∵∴∵∴∵∴∵∴|∴∵∴∵↙∵∴/10.3￣7￣
# ∴∵∴∵∴∵∴∵∴∵∴|∴∵↙∵∴∵/￣\_______
# ∴∵∴∵∴∵∴∵∴∵∴|↙∵∴18∵\__/∵∴∵∴∵∴∵
# -------------------------------

cam2lidar_len = 6.8 # cm
cam_h = 10.3 # cm
cam_angle_view_v = 120 # degree
cam_image_zero = 18 # cm

line_range = 400 # mm
line_width = 200 # mm

rslt_img_size = 400

lidar_color = (0, 255, 0)
lidar_jajucha_color = (0, 0, 255)
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

# draw imgs
def draw_lidar(theta, dist):
    theta_rad = np.deg2rad(-theta)

    x = dist * np.cos(theta_rad)
    y = dist * np.sin(theta_rad)
    
    max_dist = 5000

    scale = (rslt_img_size / 2) / max_dist

    ix = (x * scale + rslt_img_size / 2).astype(np.int32)
    iy = (y * scale + rslt_img_size / 2).astype(np.int32)

    img = np.zeros((rslt_img_size, rslt_img_size, 3), dtype=np.uint8)

    show_jajucha = False
    if show_jajucha:
        triangle_radius = 10
        center_x, center_y = rslt_img_size // 2, rslt_img_size // 2

        triangle_pts = np.array([[
            (center_x + triangle_radius, center_y),  # 오른쪽
            (center_x - triangle_radius, center_y - triangle_radius),  # 왼쪽 위
            (center_x - triangle_radius, center_y + triangle_radius)   # 왼쪽 아래
        ]], dtype=np.int32)

        cv2.fillPoly(img, triangle_pts, color=lidar_jajucha_color)  

    for nx, ny in zip(ix, iy):
        if (ix >= 0) and (ix < rslt_img_size) and (iy >= 0) and (iy < rslt_img_size):
            cv2.circle(img, (nx, ny), radius=2, color=lidar_color, thickness=-1)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
    cannyed_img = cv2.resize(cannyed_img, (cannyed_img.shape[1]/2, cannyed_img.shape[0]/2))

    roi = lidar_img[0:cannyed_img.shape[0], rslt_img_size//2 - cannyed_img.shape[1]//2:rslt_img_size//2 + cannyed_img.shape[1]//2]
    roi[np.where((cannyed_img != [0, 0, 0]).all(axis=2))] = cannyed_img[np.where((cannyed_img != [0, 0, 0]).all(axis=2))]

    lidar_img[0:cannyed_img.shape[0], rslt_img_size//2 - cannyed_img.shape[1]//2:rslt_img_size//2 + cannyed_img.shape[1]//2] = roi
    rslt_img = lidar_img
    
    # rslt_img = full_img[0:300, 0:400]

    return rslt_img