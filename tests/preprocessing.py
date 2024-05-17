from flask import Flask, request, jsonify
from time import time
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as clstr
from collections import defaultdict
from functools import partial

# CONSTS
GUESS_EDGE_SQUARE_PADDING = 15
NEUTRALIZE_ANGLE_DIFF_FACTOR = 0.69
CLASSIFY_MISSING_LINE_PADDING = 100 # +/- 100 at 2x median gap is the threshold to add a line in between them
TOO_CLOSE_TOGETHER_LINE_THRESHOLD = 90 # medium - 90 is the minimum distance between two lines, other wise one gets removed

def auto_canny(image, sigma=0.33):
    """
    Canny edge detection with automatic thresholds.
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
 
    # return the edged image
    return edged

def cluster(points, max_dist=50):
    """
    Given a list of points, returns a list of cluster centers.
    """
    Y = spatial.distance.pdist(points)
    Z = clstr.hierarchy.single(Y)
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:,0]), np.mean(np.array(arr)[:,1])), clusters)
    return clusters

def closest_point(points, loc):
    """
    Returns the list of points, sorted by distance from loc.
    """
    dists = np.array(map(partial(spatial.distance.euclidean, loc), points))
    return points[dists.argmin()]

def find_corners(points, img_dim):
    """
    Given a list of points, returns a list containing the four corner points.
    """
    center_point = closest_point(points, (img_dim[0] / 2, img_dim[1] / 2))
    points.remove(center_point)
    center_adjacent_point = closest_point(points, center_point)
    points.append(center_point)
    grid_dist = spatial.distance.euclidean(np.array(center_point), np.array(center_adjacent_point))
    
    img_corners = [(0, 0), (0, img_dim[1]), img_dim, (img_dim[0], 0)]
    board_corners = []
    tolerance = 0.25 # bigger = more tolerance
    for img_corner in img_corners:
        while True:
            cand_board_corner = closest_point(points, img_corner)
            points.remove(cand_board_corner)
            cand_board_corner_adjacent = closest_point(points, cand_board_corner)
            corner_grid_dist = spatial.distance.euclidean(np.array(cand_board_corner), np.array(cand_board_corner_adjacent))
            if corner_grid_dist > (1 - tolerance) * grid_dist and corner_grid_dist < (1 + tolerance) * grid_dist:
                points.append(cand_board_corner)
                board_corners.append(cand_board_corner)
                break
    return board_corners

def four_point_transform(img, points, square_length=1816):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [0, square_length], [square_length, square_length], [square_length, 0]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (square_length, square_length))

def find_board(fname):
    img = cv2.imdecode(fname, 1)
    if img is None:
        return [None, "no image"]

    # Merge all similar green colors in the original image like a cartoon
    lower_green = np.array([50, 50, 50])
    upper_green = np.array([100, 255, 255])
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_green, upper_green)

    # test splitting by outline 
    # lower_black = np.array([0, 0, 0])
    # upper_black = np.array([180, 255, 30])
    # mask2 = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lower_black, upper_black)
    
    img = cv2.bitwise_and(img, img, mask=mask)
    # img = cv2.bitwise_and(img, img, mask=mask2)
    # color the mask 100% blue
    img[mask > 0] = [255, 0, 0]
    # img[mask2 > 0] = [0, 255, 255]
    cv2.imwrite('mask.jpg', img)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    cv2.imwrite('gray.jpg', gray)

    # Canny edge detection
    edges = auto_canny(gray)

    cv2.imwrite('edges.jpg', edges)

    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    lines = np.reshape(lines, (-1, 2))

    # merge similar lines
    new_lines = []
    for rho, theta in lines:
        for new_rho, new_theta in new_lines:
            if abs(rho - new_rho) < 50 and abs(theta - new_theta) < np.pi / 18:
                break
        else:
            new_lines.append((rho, theta))

    vertical_lines = []
    horizontal_lines = []
    for rho, theta in new_lines:
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            vertical_lines.append((rho, theta))
        else:
            horizontal_lines.append((rho, theta))
    
    horizontal_lines = sorted(horizontal_lines, key=lambda x: x[0])
    vertical_lines = sorted(vertical_lines, key=lambda x: x[0])

    # test popping
    # vertical_lines.pop(7)
    # vertical_lines.pop(6)
    # vertical_lines.pop(1)
    # horizontal_lines.pop(0)
    # horizontal_lines.pop(1)

    # VERTICAL LINE PROCESSING
    
    # REMOVE VERTICAL LINES THAT ARE TOO CLOSE TOGETHER
    htan = (img.shape[0] / 2, np.pi / 2)
    intersections = []
    for rho, theta in vertical_lines:
        A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(htan[1]), np.sin(htan[1])]])
        b = np.array([rho, htan[0]])
        point = np.linalg.solve(A, b)
        intersections.append((point, rho, theta))

    intersections = sorted(intersections, key=lambda x: x[0][0])

    hgaps = []
    for i in range(len(intersections) - 1):
        # get distance between this interesction[0] and the next
        hgaps.append(abs(intersections[i + 1][0][0] - intersections[i][0][0]))
        
    hmedian = np.median(hgaps)

    for i in range(len(intersections) - 1):
        if i == len(intersections) - 1:
            break
        gap = abs(intersections[i + 1][0][0] - intersections[i][0][0])
        if (gap < hmedian - TOO_CLOSE_TOGETHER_LINE_THRESHOLD):
            print('removing to-close vertical', intersections[i][0][0])
            # get veritcal line that corresponds to this intersection
            for j in range(len(vertical_lines)):
                if vertical_lines[j][0] == intersections[i][1] and vertical_lines[j][1] == intersections[i][2]:
                    vertical_lines.pop(j)
                    intersections.pop(i)
                    break

    
    changesMade = False

    # INTELLIGENTLY FILL IN MISING VERTICAL LINES
    while len(vertical_lines) < 9:
        # if there is a gap between two vertical lines thats approximately twice the median gap, add a line there
        for i in range(len(intersections) - 1):
            gap = abs(intersections[i + 1][0][0] - intersections[i][0][0])
            if (gap > ((hmedian * 2) - CLASSIFY_MISSING_LINE_PADDING) and (gap < (hmedian * 2) + CLASSIFY_MISSING_LINE_PADDING)):
                # Add a new vertical line at the middle of the gap
                new_rho = (intersections[i][1] + intersections[i + 1][1]) / 2
                new_theta = (intersections[i][2] + intersections[i + 1][2]) / 2

                print('new vertical line (to fill in 2x gap) at', new_rho, new_theta)

                vertical_lines.append((new_rho, new_theta))
                # recompute intersections
                intersections = []
                for rho, theta in vertical_lines:
                    A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(htan[1]), np.sin(htan[1])]])
                    b = np.array([rho, htan[0]])
                    point = np.linalg.solve(A, b)
                    intersections.append((point, rho, theta))

                intersections = sorted(intersections, key=lambda x: x[0][0])

                changesMade = True
                break

        if changesMade:
            changesMade = False
            continue

        # we're missing an edge vertical line, so add one to whichever line is farthest from the left or right
        leftmost = intersections[0][0][0]
        rightmost = intersections[-1][0][0]
        imageleft = 0
        imageright = img.shape[1]
        # rho is the distance from the origin, so we add the median distance to the last rho
        # theta is the angle of the line, so we get the difference between the last two thetas and add that to the last theta
        if abs(leftmost - imageleft) > abs(rightmost - imageright):
            # Add a new vertical line at the left

            new_rho = intersections[0][1] - hmedian - GUESS_EDGE_SQUARE_PADDING
            theta_diff = intersections[1][2] - intersections[0][2]
            new_theta = intersections[0][2] + theta_diff * NEUTRALIZE_ANGLE_DIFF_FACTOR

            if (intersections[0][1] - hmedian - GUESS_EDGE_SQUARE_PADDING) < 0:
                print("Swapping orientation relative to leftmode")
                new_rho = intersections[0][1] + hmedian + GUESS_EDGE_SQUARE_PADDING

            print('new vertical line to left', new_rho, new_theta)
            vertical_lines.append((new_rho, new_theta))

        else:
            # Add a new vertical line at the right
            new_rho = intersections[-1][1] + hmedian + GUESS_EDGE_SQUARE_PADDING
            theta_diff = intersections[-1][2] - intersections[-2][2]
            new_theta = intersections[-1][2] + theta_diff * NEUTRALIZE_ANGLE_DIFF_FACTOR

            if (intersections[-1][1] + hmedian + GUESS_EDGE_SQUARE_PADDING) < 0:
                print("Swapping orientation relative to rightmode")
                new_rho = intersections[-1][1] - hmedian - GUESS_EDGE_SQUARE_PADDING
            
            print('new vertical line to right', new_rho, new_theta)
            vertical_lines.append((new_rho, new_theta))

        # recompute intersections
        intersections = []
        for rho, theta in vertical_lines:
            A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(htan[1]), np.sin(htan[1])]])
            b = np.array([rho, htan[0]])
            point = np.linalg.solve(A, b)
            intersections.append((point, rho, theta))

        intersections = sorted(intersections, key=lambda x: x[0][0])

    # HORIZONTAL LINE PROCESSING
    
    horizontal_lines.pop(0)
    horizontal_lines.pop(1)

    # REMOVE HORIZONTAL LINES THAT ARE TOO CLOSE TOGETHER
    vtan = (img.shape[1] / 2, 0)
    intersections = []
    for rho, theta in horizontal_lines:
        A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(vtan[1]), np.sin(vtan[1])]])
        b = np.array([rho, htan[0]])
        point = np.linalg.solve(A, b)
        intersections.append((point, rho, theta))

    intersections = sorted(intersections, key=lambda x: x[0][1])

    vgaps = []
    for i in range(len(intersections) - 1):
        # with rho and theta, get the vertical distance this intersection[0] and the next
        vgaps.append(abs(intersections[i + 1][0][1] - intersections[i][0][1]))
        
    vmedian = np.median(vgaps)

    for i in range(len(intersections) - 1):
        if i == len(intersections) - 1:
            break
        gap = abs(intersections[i + 1][0][1] - intersections[i][0][1])
        if (gap < vmedian - TOO_CLOSE_TOGETHER_LINE_THRESHOLD):
            print('removing too-close horizontal', intersections[i][0][1])
            # get vertical line that corresponds to this intersection
            for j in range(len(horizontal_lines)):
                if horizontal_lines[j][0] == intersections[i][1] and horizontal_lines[j][1] == intersections[i][2]:
                    horizontal_lines.pop(j)
                    intersections.pop(i)
                    break

    # INTELLIGENTLY FILL IN MISING HORIZONTAL LINES
    changesMade = False
    while len(horizontal_lines) < 9:
        # if there is a gap between two vertical lines thats approximately twice the median gap, add a line there
        for i in range(len(intersections) - 1):
            gap = abs(intersections[i + 1][0][1] - intersections[i][0][1])
            if (gap > ((vmedian * 2) - CLASSIFY_MISSING_LINE_PADDING) and (gap < (vmedian * 2) + CLASSIFY_MISSING_LINE_PADDING)):
    
                # Add a new horizontal line at the middle of the gap
                new_rho = (intersections[i + 1][1] + intersections[i][1]) / 2
                new_theta = (intersections[i + 1][2] + intersections[i][2]) / 2

                print('new horizontal line (to fill in 2x gap) at', new_rho, new_theta)

                horizontal_lines.append((new_rho, new_theta))

                # recompute intersections
                intersections = []
                for rho, theta in horizontal_lines:
                    A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(vtan[1]), np.sin(vtan[1])]])
                    b = np.array([rho, vtan[0]])
                    point = np.linalg.solve(A, b)
                    intersections.append((point, rho, theta))

                intersections = sorted(intersections, key=lambda x: x[0][1])

                changesMade = True
                break

        if changesMade:
            changesMade = False
            continue

        # we're missing an edge horizontal line, so add one to whichever line is farthest from the top or bottom
        uppermost = intersections[0][0][1]
        lowermost = intersections[-1][0][1]
        imagetop = 0
        imagebottom = img.shape[0]

        # rho is the distance from the origin, so we add the median distance to the last rho
        # theta is the angle of the line, so we get the difference between the last two thetas and add that to the last theta
        if abs(uppermost - imagetop) > abs(lowermost - imagebottom):
            print('new horizontal line to top')
            # Add a new horizontal line at the top
            new_rho = intersections[0][1] - hmedian - GUESS_EDGE_SQUARE_PADDING
            theta_diff = intersections[1][2] - intersections[0][2]
            new_theta = intersections[0][2] + theta_diff * NEUTRALIZE_ANGLE_DIFF_FACTOR # neuterize the angle a bit

            horizontal_lines.append((new_rho, new_theta))
        else:
            print('new horizontal line to bottom')
            # Add a new horizontal line at the bottom
            new_rho = intersections[-1][1] + hmedian + GUESS_EDGE_SQUARE_PADDING

            theta_diff = intersections[-1][2] - intersections[-2][2]
            new_theta = intersections[-1][2] + theta_diff * NEUTRALIZE_ANGLE_DIFF_FACTOR # neutralize the angle a bit

            horizontal_lines.append((new_rho, new_theta))
        
        # recompute intersections
        intersections = []
        for rho, theta in horizontal_lines:
            A = np.array([[np.cos(theta), np.sin(theta)], [np.cos(vtan[1]), np.sin(vtan[1])]])
            b = np.array([rho, vtan[0]])
            point = np.linalg.solve(A, b)
            intersections.append((point, rho, theta))

        intersections = sorted(intersections, key=lambda x: x[0][1])

    for rho, theta in vertical_lines:
        a = np.cos(theta) # this is the x component of the unit vector
        b = np.sin(theta) # this is the y component of the unit vector
        x0 = a * rho # this is the x coordinate of the point
        y0 = b * rho # this is the y coordinate of the point

        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for rho, theta in horizontal_lines:

        a = np.cos(theta) # this is the x component of the unit vector
        b = np.sin(theta) # this is the y component of the unit vector
        x0 = a * rho # this is the x coordinate of the point
        y0 = b * rho # this is the y coordinate of the point

        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * a)
        
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * a)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


    cv2.imwrite('hough_lines_on_image.jpg', img)
    
    # Compute intersection points
    # h, v = hor_vert_lines(lines)
    # if len(h) < 9 or len(v) < 9:
    #     return [None, "too few lines"]
    # points = intersections(h, v)
    
    # # Cluster intersection points
    # points = cluster(points)

    # # Find corners
    # img_shape = np.shape(img)
    # points = find_corners(points, (img_shape[1], img_shape[0]))
    
    # # Perspective transform
    # new_img = four_point_transform(img, points)

    # # write new image
    # cv2.imwrite('board.jpg', new_img)

    # return new_img
    exit()
    return 0

def hor_vert_lines(lines):
    """
    A line is given by rho and theta. Given a list of lines, returns a list of
    horizontal lines (theta=90 deg) and a list of vertical lines (theta=0 deg).
    """
    h = []
    v = []
    for distance, angle in lines:
        if angle < np.pi / 4 or angle > np.pi - np.pi / 4:
            v.append([distance, angle])
        else:
            h.append([distance, angle])
    return h, v

def intersections(h, v):
    """
    Given lists of horizontal and vertical lines in (rho, theta) form, returns list
    of (x, y) intersection points.
    """
    points = []
    for d1, a1 in h:
        for d2, a2 in v:
            A = np.array([[np.cos(a1), np.sin(a1)], [np.cos(a2), np.sin(a2)]])
            b = np.array([d1, d2])
            point = np.linalg.solve(A, b)
            points.append(point)
    return np.array(points)

def split_board(img):
    """
    Given a board image, returns an array of 64 smaller images.
    """
    arr = []
    sq_len = img.shape[0] / 8
    for i in range(8):
        for j in range(8):
            arr.append(img[i * sq_len : (i + 1) * sq_len, j * sq_len : (j + 1) * sq_len])
    return arr

if __name__ == '__main__':
    # file = open('/Users/rohin/Desktop/final.jpg', 'rb')
    file = open('/Users/rohin/Desktop/IMG_8452.jpg', 'rb')
    img = np.asarray(bytearray(file.read()))
    boards = find_board(img)
    board = boards[0]
    reason = boards[1]
    if board is None:
        print(reason)
    squares = split_board(board)
    print('finished board rec')
