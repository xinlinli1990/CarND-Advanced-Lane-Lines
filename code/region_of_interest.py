import cv2
import numpy as np

def get_RoI_vertices(image):
    # Define a convex polygon mask for the Region of Interest
    imshape = image.shape

    left_bottom_point = [np.round(0.00 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]
    left_top_point = [np.round(0.4 * imshape[1]).astype(int), np.round((0.63) * imshape[0]).astype(int)]
    right_top_point = [np.round(0.6 * imshape[1]).astype(int), np.round((0.63) * imshape[0]).astype(int)]
    right_bottom_point = [np.round(1.00 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]

    horizontal_shift = 0.0  # np.round(0.02 * imshape[1]).astype(int) #0.02
    left_bottom_point[0] += horizontal_shift
    left_top_point[0] += horizontal_shift
    right_top_point[0] += horizontal_shift
    right_bottom_point[0] += horizontal_shift

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

    return np.array([[left_bottom_point, left_top_point, right_top_point, right_bottom_point]], dtype=np.int32)


def apply_Region_of_Interest(img, vertices):
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


def draw_RoI(image, RoI_vertices, thickness=3):
    left_bottom_point = tuple(RoI_vertices[0, 0])
    left_top_point = tuple(RoI_vertices[0, 1])
    right_top_point = tuple(RoI_vertices[0, 2])
    right_bottom_point = tuple(RoI_vertices[0, 3])

    color = (0, 0, 255)

    cv2.line(image, left_bottom_point, left_top_point, color, thickness)
    cv2.line(image, left_top_point, right_top_point, color, thickness)
    cv2.line(image, right_top_point, right_bottom_point, color, thickness)
    cv2.line(image, left_bottom_point, right_bottom_point, color, thickness)

    return image