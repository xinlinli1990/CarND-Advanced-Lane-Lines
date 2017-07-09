import cv2
import glob
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import numpy as np
from outliers import smirnov_grubbs as grubbs

from image_processing_pipeline import image_process_pipeline, get_lr_points
from image_undistortion import cal_undistort, get_objpoints_imgpoints_from_chessboard_images
from image_warping import get_PerspectiveTrans_vertices, image_warp
from region_of_interest import get_RoI_vertices, apply_Region_of_Interest, draw_RoI
from sliding_window_search import polynomial_fit, heatmap_polynomial_fit


def pass_sanity_check(new_time_step, previous_time_steps):
    if len(previous_time_steps) <= 8:
        return True

    prev_curverads = []
    for time_step in previous_time_steps:
        prev_curverads.append(time_step['curverad'])

    # ploty = np.linspace(0, 719, num=2).astype(np.uint)
    #
    # print("previous_lanes")
    # for time_step in previous_time_steps:
    #     points = time_step['points']
    #     new_fitx, new_fit, new_curverad = polynomial_fit(points, ploty, return_fit=True, return_curverad=True)
    #     print('a=' + str(new_fit[0]) + ' b=' + str(new_fit[1]) + ' c='+str(new_fit[2]) + ' currad='+str(new_curverad))
    # new_fitx, new_fit, new_curverad = polynomial_fit(new_time_step['points'], ploty, return_fit=True, return_curverad=True)
    # print('new_lane')
    # print('a=' + str(new_fit[0]) + ' b=' + str(new_fit[1]) + ' c=' + str(new_fit[2]) + ' currad='+str(new_curverad))
    # print()

    prev_curverads.remove(max(prev_curverads))
    prev_curverads.remove(min(prev_curverads))
    prev_curverads = np.array(prev_curverads)

    diff = np.abs(new_time_step['curverad'] - np.mean(prev_curverads))
    threshold = 10 * np.std(prev_curverads)

    if diff > threshold:
        return False
    else:
        return True

    # if pass_sanity_check(new_time_step_left, previous_time_steps_left):
    #     while len(previous_time_steps_left) >= time_steps:
    #         previous_time_steps_left.pop(0)
    #     previous_time_steps_left.append(new_time_step_left)
    #
    # if pass_sanity_check(new_time_step_right, previous_time_steps_right):
    #     while len(previous_time_steps_right) >= time_steps:
    #         previous_time_steps_right.pop(0)
    #     previous_time_steps_right.append(new_time_step_right)


def grubbs_check(new_time_step, time_steps, num_time_steps=5):

    for time_step in time_steps:
        time_step['life'] -= 1
        if time_step['life'] <= 0:
            time_steps.remove(time_step)

    while len(time_steps) >= num_time_steps:
        time_steps.pop(0)

    # add
    time_steps.append(new_time_step)

    if len(time_steps) <= 5:
        return time_steps

    x_tops = []
    for time_step in time_steps:
        x_tops.append(time_step['x_top'])
    x_tops = np.array(x_tops)

    rm_idx = grubbs.two_sided_test_indices(x_tops, alpha=2.0)
    rm_idx = np.sort(rm_idx)[::-1]  # Ascending -> Descending, np.array -> list
    for idx in rm_idx:
        time_steps.pop(idx)

    return time_steps


def process_image(image):
    global image_undistortion_params
    global image_warping_params
    global window_search_params
    global RoI_vertices
    global left_time_steps
    global right_time_steps
    global num_time_steps
    global previous_centers

    if image_warping_params['PerspectiveTrans_vertices'] is None:
        image_warping_params['PerspectiveTrans_vertices'] = get_PerspectiveTrans_vertices(image)

    if RoI_vertices is None:
        RoI_vertices = get_RoI_vertices(image)

    # Image undistortion
    objpoints = image_undistortion_params['objpoints']
    imgpoints = image_undistortion_params['imgpoints']
    undistorted = cal_undistort(image, objpoints, imgpoints)

    # Get l_points and r_points from new image
    new_l_points, new_r_points = get_lr_points(undistorted, image_warping_params, RoI_vertices, previous_centers=previous_centers, window_search_params=window_search_params)

    #ploty = np.linspace(0, 719, num=720).astype(np.uint)
    ploty = np.linspace(0, 719, num=720).astype(np.uint)
    ploty = np.array(ploty[::-1])
    new_left_fitx, new_left_fit, new_left_curverad = polynomial_fit(new_l_points, ploty, return_fit=True, return_curverad=True)
    new_right_fitx, new_right_fit, new_right_curverad = polynomial_fit(new_r_points, ploty, return_fit=True, return_curverad=True)
    # print(new_left_fitx[0])
    # print(new_right_fitx[0])
    # print(new_left_fitx[-1])
    # print(new_right_fitx[-1])
    # if previous_centers is not None:
    #     print(previous_centers['l_center'])
    #     print(previous_centers['r_center'])

    new_time_step_left = {'points': new_l_points, 'curverad': new_left_curverad, 'life': 24, 'x_top': new_left_fitx[-1]} # 25
    new_time_step_right = {'points': new_r_points, 'curverad': new_right_curverad, 'life': 24, 'x_top': new_right_fitx[-1]}

    if new_time_step_left['x_top'] < new_time_step_right['x_top']:
        left_time_steps = grubbs_check(new_time_step_left, left_time_steps, num_time_steps)
        right_time_steps = grubbs_check(new_time_step_right, right_time_steps, num_time_steps)

    l_heatmap = np.zeros_like(new_l_points)
    r_heatmap = np.zeros_like(new_r_points)
    for time_step in left_time_steps:
        l_heatmap[time_step['points'] == 255] += 1
    for time_step in right_time_steps:
        r_heatmap[time_step['points'] == 255] += 1

    l_points = l_heatmap
    r_points = r_heatmap

    window_width = 50
    window = np.ones(window_width)
    l_sum = np.sum(l_points[int(3 * l_points.shape[0] / 4):, :int(l_points.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(r_points[int(3 * r_points.shape[0] / 4):, int(r_points.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(l_points.shape[1] / 2)

    if previous_centers is None:
        previous_centers = {}
    previous_centers['l_center'] = l_center
    previous_centers['r_center'] = r_center
    
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension


    left_fitx, left_fit, left_curverad = heatmap_polynomial_fit(l_points, 
                                                                ploty, 
                                                                xm_per_pix=xm_per_pix, 
                                                                ym_per_pix=ym_per_pix, 
                                                                return_fit=True, 
                                                                return_curverad=True)  # polynomial_fit(l_points, ploty, return_fit=True, return_curverad=True)
    right_fitx, right_fit, right_curverad = heatmap_polynomial_fit(r_points, 
                                                                   ploty, 
                                                                   xm_per_pix=xm_per_pix, 
                                                                   ym_per_pix=ym_per_pix, 
                                                                   return_fit=True, 
                                                                   return_curverad=True)
    
    curverad = 0.5 * left_curverad + 0.5 * right_curverad

    center_x_bottom = (left_fitx[0] + right_fitx[0]) / 2
    horizontal_shift_pixels = center_x_bottom - (image.shape[1]*0.5)
    horizontal_shift = horizontal_shift_pixels * xm_per_pix
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(undistorted[:,:,0]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    green_red_pos = 300
    red_pts_left = np.array([np.transpose(np.vstack([left_fitx[green_red_pos:], ploty[green_red_pos:]]))])
    red_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[green_red_pos:], ploty[green_red_pos:]])))])
    red_pts = np.hstack((red_pts_left, red_pts_right))
    green_pts_left = np.array([np.transpose(np.vstack([left_fitx[:green_red_pos], ploty[:green_red_pos]]))])
    green_pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx[:green_red_pos], ploty[:green_red_pos]])))])
    green_pts = np.hstack((green_pts_left, green_pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([green_pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([red_pts]), (255, 0, 0))

    # Unwarp color_warp
    PerspectiveTrans_vertices = image_warping_params['PerspectiveTrans_vertices']# get_PerspectiveTrans_vertices(undistorted)
    offsets = image_warping_params['offsets']
    newwarp = image_warp(color_warp, PerspectiveTrans_vertices, offsets, unwarp=True)

    # Combine color_warp and the original image
    #undistorted = draw_RoI(undistorted, RoI_vertices)
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    
    # Draw the text
    text = "Vehicle is {:0.4f} m".format(abs(horizontal_shift))
    if horizontal_shift > 0:
        text += " right of the center"
    else:
        text += " left of the center"
    
    cv2.putText(result, text=text, org=(int(result.shape[1]*0.05), int(result.shape[0]*0.1)), fontFace=cv2.FONT_ITALIC, fontScale=1, color=(255,255,255)) #FONT_HERSHEY_SIMPLEX
    text = "Radius of curvature: {:5d} m".format(int(curverad))  #+ str(curverad) + "m"
    cv2.putText(result, text=text, org=(int(result.shape[1]*0.05), int(result.shape[0]*0.15)), fontFace=cv2.FONT_ITALIC, fontScale=1, color=(255,255,255))
    
    image = result

    return image


f_paths = ['../project_video_detected.mp4',
           # '../challenge_video.mp4',
           # '../harder_challenge_video.mp4',
           ]

output_paths = ['./tmp.mp4',
                # './challenge_video_test2.mp4',
                # './harder_challenge_video_test2.mp4',
                ]

# f_paths = ['./cut.mp4',
#            ]
#
# output_paths = ['./project_video_test2.mp4',
#                 ]

chessboard_image_paths = glob.glob('../camera_cal/calibration*.jpg')

objpoints, imgpoints = get_objpoints_imgpoints_from_chessboard_images(chessboard_image_paths, 9, 6)

image_undistortion_params = {'objpoints': objpoints, 'imgpoints': imgpoints}

num_time_steps = 12  # 15

PerspectiveTrans_vertices = None # get_PerspectiveTrans_vertices(undistorted)
offsets = {'vertical_offset': 655, 'horizontal_offset': 500}  # 680, 500

RoI_vertices = None

image_warping_params = {
    'PerspectiveTrans_vertices': PerspectiveTrans_vertices,
    'offsets': offsets
}

window_search_params = {
    'window_width': 50,
    'window_height': 40,  # Break image into 9 vertical layers since image height is 720 #80
    'margin': 100,  # How much to slide left and right for searching
}

left_time_steps = []
right_time_steps = []
previous_centers = None

# img = cv2.imread("../test_images/test1.jpg")  # project_video2.jpg
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# img = process_image(img)
# plt.imshow(img)
# plt.show()

for f_path, output_path in zip(f_paths, output_paths):
    left_time_steps = []
    right_time_steps = []

    clip = VideoFileClip(f_path)
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_path, audio=False)