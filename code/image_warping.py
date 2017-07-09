import cv2
import numpy as np

def get_PerspectiveTrans_vertices(image):
    # Define a convex polygon mask for the Region of Interest
    imshape = image.shape

    left_bottom_point = [np.round(0.23 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]
    left_top_point = [np.round(0.355 * imshape[1]).astype(int), np.round((0.8) * imshape[0]).astype(int)]
    right_top_point = [np.round(0.68 * imshape[1]).astype(int), np.round((0.8) * imshape[0]).astype(int)]
    right_bottom_point = [np.round(0.845 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]

    # Horizontal shift
    horizontal_shift = np.round(0.005 * imshape[1]).astype(int)  # 0.02
    left_bottom_point[0] += horizontal_shift
    left_top_point[0] += horizontal_shift
    right_top_point[0] += horizontal_shift
    right_bottom_point[0] += horizontal_shift

    # Vertical shift
    vertical_shift = -1 * np.round(0.025 * imshape[0]).astype(int)
    left_bottom_point[1] += vertical_shift
    left_top_point[1] += vertical_shift
    right_top_point[1] += vertical_shift
    right_bottom_point[1] += vertical_shift

    # Convert to tuple
    left_bottom_point = tuple(left_bottom_point)
    left_top_point = tuple(left_top_point)
    right_top_point = tuple(right_top_point)
    right_bottom_point = tuple(right_bottom_point)

    return np.array([[left_top_point, right_top_point, right_bottom_point, left_bottom_point]], dtype=np.int32)

def image_warp(image, PerspectiveTrans_vertices, offsets, unwarp=False):

    img_size = image.shape[0:2][::-1]

    vertical_offset = offsets['vertical_offset']
    horizontal_offset = offsets['horizontal_offset']

    src = np.float32(PerspectiveTrans_vertices)
    dst = np.float32([[horizontal_offset, vertical_offset], [img_size[0] - horizontal_offset, vertical_offset],
                      [img_size[0] - horizontal_offset, img_size[1]], [horizontal_offset, img_size[1]]])

    if unwarp:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(image, M, img_size)

    return warped