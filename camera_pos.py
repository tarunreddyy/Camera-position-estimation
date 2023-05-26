import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import sleep

def ret_images(file):
    """
    Convert video to frames
    input: file path
    output: image list
    """
    # Images list
    images = []
    # Load video
    cap = cv2.VideoCapture(file)
    # Loop through frames of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to HSV color space
        images.append(frame)
    # Release the video
    cap.release()
    return images

def detect_paper(images):
    """
    Detects paper in the images by creating binary image masking surrounding 
    input: image list 
    output: binary image list
    """
    # Define the lower and upper bounds for the white color in HSV color space
    white_lower = np.array([0,0,185])
    white_upper = np.array([180,111,255])
    paper_imgs = []

    for i in range(len(images)):
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(images[i], cv2.COLOR_BGR2HSV)

        # Threshold the image to get the white pixels
        mask = cv2.inRange(hsv, white_lower, white_upper)

        # Remove small contours and keep the largest
        labels, num_labels = ndimage.label(mask)
        sizes = ndimage.sum(mask, labels, range(1, num_labels+1))
        largest_label = np.argmax(sizes) + 1
        new_mask = np.zeros_like(mask)
        new_mask[labels == largest_label] = 1
        # Blur the image for better edge detection
        img_blur = cv2.medianBlur(new_mask, 7) * 255
        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)

        paper_imgs.append(edges)
    return paper_imgs

def hough_transform(edge_map):
    """
    Detects lines using Hough Transform
    input: edge map of only the paper
    output: lines detected in the image
    """
    # Set parameters for Hough transform
    theta_res = 1 # Angular resolution (in degrees)
    rho_res = 1 # Distance resolution (in pixels)
    threshold = 118 # Minimum number of votes to consider a line

    # Define range of theta and rho
    num_theta = int(180 / theta_res)
    num_rho = int(np.sqrt(edge_map.shape[0]**2 + edge_map.shape[1]**2) / rho_res)
    theta_range = np.linspace(-90, 89, num_theta)
    rho_range = np.linspace(-num_rho*rho_res, num_rho*rho_res, 2*num_rho+1)

    # Initialize accumulator array
    accumulator = np.zeros((2*num_rho+1, num_theta))

    # Compute Hough transform
    for y in range(edge_map.shape[0]):
        for x in range(edge_map.shape[1]):
            if edge_map[y,x] > 0: # Check if pixel is an edge
                for i,theta in enumerate(theta_range):
                    rho = x*np.cos(np.deg2rad(theta)) + y*np.sin(np.deg2rad(theta))
                    j = np.argmin(np.abs(rho_range - rho))
                    accumulator[j,i] += 1

    # Find local maxima in accumulator
    lines = []
    for j in range(1, accumulator.shape[0]-1):
        for i in range(1, accumulator.shape[1]-1):
            if accumulator[j,i] > threshold and accumulator[j,i] > accumulator[j-1,i-1] and accumulator[j,i] > accumulator[j+1,i+1] and accumulator[j,i] > accumulator[j-1,i+1] and accumulator[j,i] > accumulator[j+1,i-1]:
                theta = theta_range[i]
                rho = rho_range[j]
                lines.append((rho, theta))
    return lines

def find_points(lines):
    """
    Detects corners from the lines
    input: lines detected from the hough transform
    output: corners (intersection points)
    """
    intersections = []
    for i in range(len(lines)):
        rho1, theta1 = lines[i]
        a1 = np.cos(np.deg2rad(theta1))
        b1 = np.sin(np.deg2rad(theta1))
        for j in range(i+1, len(lines)):
            rho2, theta2 = lines[j]
            a2 = np.cos(np.deg2rad(theta2))
            b2 = np.sin(np.deg2rad(theta2))
            det = a1*b2 - a2*b1
            if np.abs(det) > 0.001:
                x = (b2*rho1 - b1*rho2) / det
                y = (-a2*rho1 + a1*rho2) / det
                intersections.append((x,y))

    if any(x < 0 or y < 0 for x, y in intersections):
        # Remove negative numbers
        intersections = [(x, y) for x, y in intersections if x >= 0 and y >= 0]

    new_points = []
    threshold = 2
    for i in range(len(intersections)):
        keep = True
        for j in range(i+1, len(intersections)):
            distance = ((intersections[i][0] - intersections[j][0])**2 + (intersections[i][1] - intersections[j][1])**2)**0.5
            if distance < threshold:
                keep = False
                break
        if keep:
            new_points.append(intersections[i])
    
    return new_points

def compute_homography(points):
    """
    Computes homography for the corners from the image and world points
    input: corners from the image
    output: homography matrix
    """
    # Define the real world coordinates of the A4 paper
    paper_points = np.array([[0, 0],[216, 0],[216, 279],[0, 279]], dtype=np.float32)

    # Define the image coordinates of the four corners of the A4 paper
    image_points = np.array(points, dtype=np.float32)

    # Compute the homography matrix using DLT algorithm
    A = []
    for i in range(4):
        x, y = image_points[i]
        u, v = paper_points[i]
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape((3, 3))
    H = H / H[2, 2]

    return H

def compute_camera_matrix(H,K):
    """
    Computes camera matrix from homography matrix and intrinsic matrix
    input: homography matrix and intrinsic matrix
    output: camera matrix
    """
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]
    lam = 1 / np.linalg.norm(np.dot(K_inv, h1))

    r1 = lam * np.dot(K_inv, h1)
    r2 = lam * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)

    t = lam * np.dot(K_inv, h3)

    R = np.column_stack((r1, r2, r3))

    return np.column_stack((np.dot(K, R), t))

def extract_camera_pos(camera_matrix):
    """
    Extracting camera position from the camera matrix
    input: camera matrix
    output: camera position, roll, pitch, yaw
    """
    camera_pos = -np.dot(np.linalg.inv(camera_matrix[:, :-1]), camera_matrix[:, -1])
    R = camera_matrix[:3, :3]
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return camera_pos, roll, pitch, yaw


#########################################################
# Plot functions
#########################################################
def plot_camera_position(pos, pitch, roll, yaw):
    paper_corners = np.array([[0, 0, 0], [21.6, 0, 0], [21.6, 27.9, 0], [0, 27.9, 0]])

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    # plot 3D points
    xs, ys, zs = zip(*pos)
    z = [abs(num) for num in zs]
    ax1.set_xlim(-300, 300)
    ax1.set_ylim(-200, 200)
    ax1.set_zlim(0, max(z))
    ax1.scatter(xs, ys, z, c='r', marker='o')
    # Plot paper
    for i in range(4):
        ax1.plot([paper_corners[i, 0], paper_corners[(i+1)%4, 0]],
                [paper_corners[i, 1], paper_corners[(i+1)%4, 1]],
                [paper_corners[i, 2], paper_corners[(i+1)%4, 2]],
                c='b')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot second list of points in 2D
    ax2.plot(range(len(pitch)), pitch, c='g', marker='o')
    ax2.set_xlabel('frames')
    ax2.set_ylabel('Pitch')

    # Plot third list of points in 2D
    ax3.plot(range(len(roll)), roll, c='b', marker='o')
    ax3.set_xlabel('frames')
    ax3.set_ylabel('Roll')

    # Plot fourth list of points in 2D
    ax4.plot(range(len(yaw)), yaw, c='y', marker='o')
    ax4.set_xlabel('frames')
    ax4.set_ylabel('Yaw')

    plt.show()            

def plot_lines(edge_map,lines):
    # Plot edge map and Hough lines
    plt.figure(figsize=(15,10))
    plt.imshow(edge_map, cmap='gray')
    ax = plt.gca()
    for rho, theta in lines:
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        ax.plot([x1, x2], [y1, y2], color='r')
    plt.axis('off')
    plt.show()

def plot_points_on_image(img, points):
    plt.figure(figsize=(10,10))
    plt.imshow(img,cmap='gray')
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x, y, color='red')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # Load images
    All_images = ret_images("resources/project2.avi")
    # Defining K matrix
    K = np.array([[1.38E+03, 0, 9.46E+02],
                  [0, 1.38E+03, 5.27E+02],
                  [0, 0, 1]])
    Camera_poses = []
    pitch = []
    roll = []
    yaw = []

    # Preprocess Images
    paper_imgs = detect_paper(All_images)

    for i in tqdm(range(len(paper_imgs)), ncols = 100, desc ="Images"):
        lines = hough_transform(paper_imgs[i])
        corners = find_points(lines)
        if len(corners) == 4:
            # compute homography
            H = compute_homography(corners)
            # compute camera matrix
            camera_matrix = compute_camera_matrix(H, K)
            pos, r, p, y = extract_camera_pos(camera_matrix)
            Camera_poses.append(pos)
            pitch.append(p)
            roll.append(r)
            yaw.append(y)
        else:
            continue
        sleep(.1)

    plot_camera_position(Camera_poses, pitch, roll, yaw)