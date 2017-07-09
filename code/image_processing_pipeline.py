import cv2
import numpy as np
import matplotlib.pyplot as plt

from region_of_interest import get_RoI_vertices, apply_Region_of_Interest, draw_RoI
from image_undistortion import get_objpoints_imgpoints_from_chessboard_images, cal_undistort
from image_warping import get_PerspectiveTrans_vertices, image_warp
from sliding_window_search import window_mask, find_window_centroids, apply_window_centroids_mask, polynomial_fit

def apply_color_threshold(img, color_space_conversion, color_thres):
    C012 = cv2.cvtColor(img, color_space_conversion)

    C0 = C012[:, :, 0]
    C1 = C012[:, :, 1]
    C2 = C012[:, :, 2]

    C0_binary_output = np.zeros_like(C0)
    C1_binary_output = np.zeros_like(C1)
    C2_binary_output = np.zeros_like(C2)

    C0_binary_output[(C0 > color_thres[0][0]) & (C0 < color_thres[0][1])] = 1
    C1_binary_output[(C1 > color_thres[1][0]) & (C1 < color_thres[1][1])] = 1
    C2_binary_output[(C2 > color_thres[2][0]) & (C2 < color_thres[2][1])] = 1

    return C0_binary_output, C1_binary_output, C2_binary_output

def image_process_pipeline(image, image_undistortion_params, previous_centers=None):
    # Input RGB

    # Image undistortion
    objpoints = image_undistortion_params['objpoints']
    imgpoints = image_undistortion_params['imgpoints']

    RoI_vertices = get_RoI_vertices(image)

    undistorted = cal_undistort(image, objpoints, imgpoints)

    # Apply color space thresold
    LAB_space = cv2.COLOR_RGB2LAB
    LAB_thres = [[210, 255], [0, 255], [150, 255]]  # lower and upper bounds for L, A and B channel
    L_binary_output, A_binary_output, B_binary_output = apply_color_threshold(undistorted, LAB_space, LAB_thres)

    # Apply region of interest
    L_binary_output = apply_Region_of_Interest(L_binary_output, RoI_vertices)
    B_binary_output = apply_Region_of_Interest(B_binary_output, RoI_vertices)

    # Combine L and B binary output as final output
    bin_output = np.zeros_like(L_binary_output)
    bin_output[(L_binary_output == 1) | (B_binary_output == 1)] = 1

    # # Warped
    PerspectiveTrans_vertices = get_PerspectiveTrans_vertices(undistorted)
    #
    offsets = {
        'vertical_offset': 670,
        'horizontal_offset': 500
    }
    warped = image_warp(bin_output, PerspectiveTrans_vertices, offsets)

    # return np.array(cv2.merge((warped*255,warped*255,warped*255)),np.uint8)

    # find window centroids
    window_search_param = {
        'window_width': 50,
        'window_height': 80,  # Break image into 9 vertical layers since image height is 720
        'margin': 100,  # How much to slide left and right for searching
    }

    window_centroids = find_window_centroids(warped, previous_centers=previous_centers, **window_search_param)

    l_points, r_points = apply_window_centroids_mask(warped, window_centroids, **window_search_param)

    ploty = np.linspace(0, 719, num=720).astype(np.uint)
    left_fitx, left_fit = polynomial_fit(l_points, ploty, return_fit=True)
    right_fitx, right_fit = polynomial_fit(r_points, ploty, return_fit=True)

    # for x, y in zip(left_fitx, ploty):
    #     cv2.circle(undistorted, (np.int_(x), y), 1, (255, 0, 0))
    #
    # for x, y in zip(right_fitx, ploty):
    #     cv2.circle(undistorted, (np.int_(x), y), 1, (0, 0, 255))

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Unwarp color_warp
    newwarp = image_warp(color_warp, PerspectiveTrans_vertices, offsets, unwarp=True)

    # Combine color_warp and the original image
    undistorted = draw_RoI(undistorted, RoI_vertices)
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    image = result

    return image

def get_lr_points(undistorted, image_warping_params, RoI_vertices, window_search_params=None, previous_centers=None):
    # Apply color space thresold
    LAB_space = cv2.COLOR_RGB2LAB
    LAB_thres = [[210, 255], [0, 255], [150, 255]]  # lower and upper bounds for L, A and B channel
    L_binary_output, A_binary_output, B_binary_output = apply_color_threshold(undistorted, LAB_space, LAB_thres)

    # Apply region of interest
    L_binary_output = apply_Region_of_Interest(L_binary_output, RoI_vertices)
    B_binary_output = apply_Region_of_Interest(B_binary_output, RoI_vertices)

    # Combine L and B binary output as final output
    bin_output = np.zeros_like(L_binary_output)
    bin_output[(L_binary_output >= 1) | (B_binary_output >= 1)] = 1

    # # Warped
    PerspectiveTrans_vertices = image_warping_params['PerspectiveTrans_vertices']# get_PerspectiveTrans_vertices(undistorted)
    offsets = image_warping_params['offsets']

    warped = image_warp(bin_output, PerspectiveTrans_vertices, offsets)

    # return np.array(cv2.merge((warped*255,warped*255,warped*255)),np.uint8)

    # find window centroids
    if window_search_params is None:
        window_search_params = {
            'window_width': 50,
            'window_height': 40,  # Break image into 9 vertical layers since image height is 720 #80
            'margin': 100,  # How much to slide left and right for searching
        }

    window_centroids = find_window_centroids(warped, previous_centers=previous_centers, **window_search_params)

    l_points, r_points = apply_window_centroids_mask(warped, window_centroids, **window_search_params)

    return l_points, r_points

# import glob
# chessboard_image_paths = glob.glob('../camera_cal/calibration*.jpg')
#
# objpoints, imgpoints = get_objpoints_imgpoints_from_chessboard_images(chessboard_image_paths, 9, 6)
#
# image_undistortion_params = {'objpoints': objpoints, 'imgpoints': imgpoints}
#
# img = cv2.imread("../test_images/test1.jpg")  # project_video2.jpg
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img = image_process_pipeline(img, image_undistortion_params)
# plt.imshow(img, cmap='gray')
# plt.show()
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))