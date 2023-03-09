# import other necessary libaries
from utils import create_line, create_mask
import cv2
import numpy as np
import skimage

def hough_transform(img):
    # Define range of rho and theta values
    thetas = np.deg2rad(np.arange(-90, 90))
    diag_len = int(np.ceil(np.sqrt(img.shape[0]**2 + img.shape[1]**2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Initialize accumulator
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)

    # Loop over the edge map
    edge_y, edge_x = np.nonzero(img)
    for i in range(len(edge_x)):
        x = edge_x[i]
        y = edge_y[i]
        for j in range(len(thetas)):
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j]) + diag_len)
            accumulator[rho, j] += 1

    return accumulator, rhos, thetas

def non_max_suppression(hough_space, threshold, nhood_size=5):
    # Find peaks above threshold
    peaks = []
    for i in range(len(hough_space)):
        for j in range(len(hough_space[0])):
            if hough_space[i][j] > threshold:
                peaks.append((i, j))

    # Sort peaks by hough_space value
    peaks.sort(key=lambda x: hough_space[x[0], x[1]], reverse=True)

    # Apply non-maximum suppression
    output = []
    for peak in peaks:
        # Check if current peak is too close to already selected peaks
        too_close = False
        for output_peak in output:
            if np.sqrt((peak[0] - output_peak[0])**2 + (peak[1] - output_peak[1])**2) < nhood_size:
                too_close = True
                break
        if not too_close:
            output.append(peak)

    return output

# load the input image
img = cv2.imread("road.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# run Canny edge detector to find edge points
edges = cv2.Canny(gray, 110, 50)

# create a mask for ROI by calling create_mask
mask = create_mask(edges.shape[0], edges.shape[1])

# extract edge points in ROI by multipling edge map with the mask
masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

# Define the Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 35
min_line_len = 50
max_line_gap = 100

# Run Hough on edge detected image
result = img
hough_space, rhos, thetas = skimage.transform.hough_line(masked_edges)
_, angles, vector = skimage.transform.hough_line_peaks(hough_space, rhos, thetas, num_peaks=1, min_distance=25)
theta = angles[0]
rho = vector[0]
# Add blue line
x, y = create_line(rho, theta, result)
cv2.line(result, (x[0], y[0]), (x[-1], y[-1]), (255, 0, 0), 5)

# Find peaks in Hough space using non-maximum suppression
peaks = non_max_suppression(hough_space, threshold)

# Draw orange line
orange_line = []
_ ,rhos, thetas = hough_transform(masked_edges)
for peak in peaks:
    r = rhos[peak[0]]
    t = thetas[peak[1]]
    xs, ys = create_line(r, t, result)
    if r > 0 and t > 0.1 and t < 1.4:
        orange_line.append((xs[0], ys[0]))
        orange_line.append((xs[-1], ys[-1]))

cv2.line(result, orange_line[0], orange_line[-1], (0, 165, 255), 5)

cv2.imshow("Edges", edges)

cv2.imshow("Mask", masked_edges)

cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()


