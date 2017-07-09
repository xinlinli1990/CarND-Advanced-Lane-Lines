import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

from moviepy.editor import VideoFileClip
from IPython.display import HTML

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


chessboard_image_paths = glob.glob('../camera_cal/calibration*.jpg')

test_image_paths = []

test_image_paths.extend(glob.glob('../test_images/straight_lines*.jpg'))
test_image_paths.extend(glob.glob('../test_images/test*.jpg'))

def draw_RoI(image, RoI_vertices):
    
    left_bottom_point = tuple(RoI_vertices[0, 0])
    left_top_point = tuple(RoI_vertices[0, 1])
    right_top_point = tuple(RoI_vertices[0, 2])
    right_bottom_point = tuple(RoI_vertices[0, 3])
    
    color = (0, 0, 255)
    thickness = 3
    
    cv2.line(image, left_bottom_point, left_top_point, color, thickness)
    cv2.line(image, left_top_point, right_top_point, color, thickness)
    cv2.line(image, right_top_point, right_bottom_point, color, thickness)
    cv2.line(image, left_bottom_point, right_bottom_point, color, thickness)
    
    return image

def get_RoI_vertices(image):
    # Define a convex polygon mask for the Region of Interest
    imshape = image.shape
    
    left_bottom_point = [np.round(0.00 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]
    left_top_point = [np.round(0.4*imshape[1]).astype(int), np.round((0.63)*imshape[0]).astype(int)]
    right_top_point = [np.round(0.6*imshape[1]).astype(int), np.round((0.63)*imshape[0]).astype(int)]
    right_bottom_point = [np.round(1.00 * imshape[1]).astype(int), np.round(0.97 * imshape[0]).astype(int)]
    
    horizontal_shift = 0.0#np.round(0.02 * imshape[1]).astype(int) #0.02
    left_bottom_point[0] += horizontal_shift
    left_top_point[0] += horizontal_shift
    right_top_point[0] += horizontal_shift
    right_bottom_point[0] += horizontal_shift
    
    vertical_shift = -1 * np.round(0.025 * imshape[0]).astype(int)
    left_bottom_point[1] += vertical_shift
    left_top_point[1] += vertical_shift
    right_top_point[1] += vertical_shift
    right_bottom_point[1] += vertical_shift    
    
    #Convert to tuple
    left_bottom_point = tuple(left_bottom_point)
    left_top_point = tuple(left_top_point)
    right_top_point = tuple(right_top_point)
    right_bottom_point = tuple(right_bottom_point)
    
    return np.array([[left_bottom_point, left_top_point, right_top_point, right_bottom_point]], dtype=np.int32)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
    
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    else:
        return None
    
    abs_sobel = np.abs(sobel)
    scaled_sobel = np.uint8(255 * (abs_sobel / np.max(abs_sobel)))
    
    return scaled_sobel
    #binary_output = np.zeros_like(scaled_sobel)
    #binary_output[(scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)] = 1
    #return binary_output

def cal_undistort(img, objpoints, imgpoints):
    """
    A function that takes an image, object points, and image points
    performs the camera calibration, image distortion correction and 
    returns the undistorted image
    """
    ret, mtx, dist, rve, tve = cv2.calibrateCamera(objpoints, imgpoints, img.shape[0:2], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
    
    
# prepare object points
nx = 9
ny = 6

objpoint = np.zeros((nx*ny, 3), np.float32)
objpoint[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Colloect objpoints and imgpoints from all given chessboard images
objpoints = []
imgpoints = []

for fname in chessboard_image_paths:
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
       
    if ret is True:
        objpoints.append(objpoint)
        imgpoints.append(corners)
      
def process_image(img):

    global R_median_hist
    global G_median_hist
    global B_median_hist
    global H_median_hist
    global L_median_hist
    global S_median_hist
    global L2_median_hist
    global A2_median_hist
    global B2_median_hist
    global Y_median_hist
    global U_median_hist
    global V_median_hist
    

    undistorted = cal_undistort(img, objpoints, imgpoints)
    
    RoI_vertices = get_RoI_vertices(img)
    
    R = undistorted[:,:,0]
    G = undistorted[:,:,1]
    B = undistorted[:,:,2]
    R_RoI = region_of_interest(R, RoI_vertices)
    G_RoI = region_of_interest(G, RoI_vertices)
    B_RoI = region_of_interest(B, RoI_vertices)
    
    R_median_hist.append(np.median(R_RoI[np.nonzero(R_RoI)]))
    G_median_hist.append(np.median(G_RoI[np.nonzero(G_RoI)]))
    B_median_hist.append(np.median(B_RoI[np.nonzero(B_RoI)]))
    
    
    
    LAB = cv2.cvtColor(undistorted, cv2.COLOR_RGB2LAB)
    L2 = LAB[:,:,0]
    A2 = LAB[:,:,1]
    B2 = LAB[:,:,2]
    L_RoI2 = region_of_interest(L2, RoI_vertices)
    A_RoI2 = region_of_interest(A2, RoI_vertices)
    B_RoI2 = region_of_interest(B2, RoI_vertices)
    
    L2_median_hist.append(np.median(L_RoI2[np.nonzero(L_RoI2)]))
    A2_median_hist.append(np.median(A_RoI2[np.nonzero(A_RoI2)]))
    B2_median_hist.append(np.median(B_RoI2[np.nonzero(B_RoI2)]))

    HLS = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    H = HLS[:,:,0]
    L = HLS[:,:,1]
    S = HLS[:,:,2]
    H_RoI = region_of_interest(H, RoI_vertices)
    L_RoI = region_of_interest(L, RoI_vertices)
    S_RoI = region_of_interest(S, RoI_vertices)
    
    H_median_hist.append(np.median(H_RoI[np.nonzero(H_RoI)]))
    L_median_hist.append(np.median(L_RoI[np.nonzero(L_RoI)]))
    S_median_hist.append(np.median(S_RoI[np.nonzero(S_RoI)]))
    
    YUV = cv2.cvtColor(undistorted, cv2.COLOR_RGB2YUV)
    Y = YUV[:,:,0]
    U = YUV[:,:,1]
    V = YUV[:,:,2]
    Y_RoI = region_of_interest(Y, RoI_vertices)
    U_RoI = region_of_interest(U, RoI_vertices)
    V_RoI = region_of_interest(V, RoI_vertices)
    
    Y_median_hist.append(np.median(Y_RoI[np.nonzero(Y_RoI)]))
    U_median_hist.append(np.median(U_RoI[np.nonzero(U_RoI)]))
    V_median_hist.append(np.median(V_RoI[np.nonzero(V_RoI)]))
    
    # Gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    # sobel_x = abs_sobel_thresh(Gray, orient='x')
    # sobel_y = abs_sobel_thresh(Gray, orient='y')
    # sobel_x_RoI = region_of_interest(sobel_x, RoI_vertices)
    # sobel_y_RoI = region_of_interest(sobel_y, RoI_vertices)
    
    return img

f_paths = ['../project_video.mp4',
           '../harder_challenge_video.mp4',
           '../challenge_video.mp4',
           ]
           
output_paths = ['./project_video.jpg',
           './harder_challenge_video.jpg',
           './challenge_video.jpg',
           ]

for f_path, output_path in zip(f_paths, output_paths):

    R_median_hist = []
    G_median_hist = []
    B_median_hist = []
    H_median_hist = []
    L_median_hist = []
    S_median_hist = []
    L2_median_hist = []
    A2_median_hist = []
    B2_median_hist = []
    Y_median_hist = []
    U_median_hist = []
    V_median_hist = []
    

    output = 'tmp.mp4'
    clip = VideoFileClip(f_path)
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output, audio=False)

    plt.close('all')
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, figsize=(15, 15))
    ax1.plot(Y_median_hist)
    ax1.set_ylim([0, 255])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Median of intensity values")
    ax1.set_title('Y Image', fontsize=20)
    ax2.plot(U_median_hist)
    ax2.set_ylim([0, 255])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Median of intensity values")
    ax2.set_title('U Image', fontsize=20)
    ax3.plot(B_median_hist)
    ax3.set_ylim([0, 255])
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Median of intensity values")
    ax3.set_title('V Image', fontsize=20)

    ax4.plot(R_median_hist)
    ax4.set_ylim([0, 255])
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Median of intensity values")
    ax4.set_title('R Image', fontsize=20)
    ax5.plot(G_median_hist)
    ax5.set_ylim([0, 255])
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Median of intensity values")
    ax5.set_title('G Image', fontsize=20)
    ax6.plot(B_median_hist)
    ax6.set_ylim([0, 255])
    ax6.set_xlabel("Time")
    ax6.set_ylabel("Median of intensity values")
    ax6.set_title('B Image', fontsize=20)

    ax7.plot(H_median_hist)
    ax7.set_ylim([0, 255])
    ax7.set_xlabel("Time")
    ax7.set_ylabel("Median of intensity values")
    ax7.set_title('H Image', fontsize=20)
    ax8.plot(L_median_hist)
    ax8.set_ylim([0, 255])
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Median of intensity values")
    ax8.set_title('L Image', fontsize=20)
    ax9.plot(S_median_hist)
    ax9.set_ylim([0, 255])
    ax9.set_xlabel("Time")
    ax9.set_ylabel("Median of intensity values")
    ax9.set_title('S Image', fontsize=20)

    ax10.plot(L2_median_hist)
    ax10.set_ylim([0, 255])
    ax10.set_xlabel("Time")
    ax10.set_ylabel("Median of intensity values")
    ax10.set_title('L Image', fontsize=20)
    ax11.plot(A2_median_hist)
    ax11.set_ylim([0, 255])
    ax11.set_xlabel("Time")
    ax11.set_ylabel("Median of intensity values")
    ax11.set_title('A Image', fontsize=20)
    ax12.plot(B2_median_hist)
    ax12.set_ylim([0, 255])
    ax12.set_xlabel("Time")
    ax12.set_ylabel("Median of intensity values")
    ax12.set_title('B Image', fontsize=20)

    f.tight_layout()
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.1)
    f.savefig(output_path)