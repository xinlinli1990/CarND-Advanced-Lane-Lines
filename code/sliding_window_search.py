import cv2
import numpy as np


def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


def find_window_centroids(warped, window_width, window_height, margin, previous_centers=None):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane
    # by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    if previous_centers is None:
        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
        r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(warped.shape[1] / 2)
    else:
        l_center = previous_centers['l_center']
        r_center = previous_centers['r_center']

    #     print(np.argmax(np.convolve(window,l_sum)))
    #     print("left center: {}, right center: {}".format(l_center, r_center))
    #     plt.plot(np.arange(0, np.convolve(window, l_sum).shape[0]), np.convolve(window, l_sum))
    #     plt.show()
    #     plt.plot(np.arange(0, np.convolve(window, r_sum).shape[0]), np.convolve(window, r_sum))
    #     plt.show()

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, warped.shape[1]))
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, warped.shape[1]))

        # Only update window center position when window not empty
        if np.max(conv_signal[l_min_index:l_max_index]) != 0:
            l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        if np.max(conv_signal[r_min_index:r_max_index]) != 0:
            r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset

        if abs(l_center - r_center) <= 1.2*margin:
            break

        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids


def apply_window_centroids_mask(warped, window_centroids, window_width, window_height, margin):
    # Points used to draw all the left and right windows
    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_mask == 1) & (warped == 1)] = 255
            r_points[(r_mask == 1) & (warped == 1)] = 255

    return l_points, r_points


def polynomial_fit(points, ploty, return_fit=False, return_curverad=False):
    y_eval = np.max(ploty)
    x = np.argwhere(points == 255)[:, 1]
    y = np.argwhere(points == 255)[:, 0]

    # Fit a second order polynomial to pixel positions in each lane line
    fit = np.polyfit(y, x, 2)

    if not return_fit and not return_curverad:
        return fit_x

    fit_x = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    if return_fit and not return_curverad:
        return fit_x, fit

    if not return_fit and return_curverad:
        return fit_x, curverad

    return fit_x, fit, curverad

def heatmap_polynomial_fit(heatmap, ploty, xm_per_pix=1, ym_per_pix=1, return_fit=False, return_curverad=False):

    y_eval = np.max(ploty)
    heat_xs = np.argwhere(heatmap > 0)[:, 1].tolist()
    heat_ys = np.argwhere(heatmap > 0)[:, 0].tolist()

    x = []
    y = []

    for heat_x, heat_y in zip(heat_xs, heat_ys):
        heat = heatmap[heat_y, heat_x]
        for i in range(heat):
            x.append(heat_x)
            y.append(heat_y)
    x = np.array(x)
    y = np.array(y)

    # Fit a second order polynomial to pixel positions in each lane line
    fit = np.polyfit(y, x, 2)

    if not return_fit and not return_curverad:
        return fit_x

    fit_x = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    
    
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    if return_fit and not return_curverad:
        return fit_x, fit

    if not return_fit and return_curverad:
        return fit_x, curverad

    return fit_x, fit, curverad
