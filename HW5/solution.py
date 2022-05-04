import numpy as np
import cv2
import math
import random


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    # START
    # the following is just a placeholder to show you the output format
    matched_pairs = []
    # dot 연산을 통해 descriptor1의 각 벡터와 descriptor 2의 벡터에 대해서 모두 내적한 행렬을 생성한다.
    # 모든 element에 arc cosine를 적용해 angle을 생성한다.
    angleTable = np.arccos(np.dot(descriptors1, descriptors2.T))
    for index1, angleArray in enumerate(angleTable):
        # vector1과 angle이 가장 작은 vector2를 찾는다.
        sortedAngleArray = list(enumerate(angleArray))
        sortedAngleArray.sort(key=lambda t: t[1])
        matchSt = sortedAngleArray[0]
        matchNd = sortedAngleArray[1]
        # best match와 second match의 angle ratio가 threshold보다 작을 때만 선택한다.
        ratio = np.abs(matchSt[1] / matchNd[1])
        if ratio <= threshold:
            matched_pairs.append([index1, matchSt[0]])
    # END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)

    # START
    h_points = []
    for xy_point in xy_points:
        h_point = np.append(xy_point, 1)
        h_points.append(h_point)
    h_points = np.array(h_points)

    xy_points_out = []
    for h_point in h_points:
        xy_point_out = h.dot(h_point)
        nomalizer = xy_point_out[2] if (xy_point_out[2] != 0) else 1e-10
        xy_point_out = xy_point_out[:2] / nomalizer
        xy_points_out.append(xy_point_out)
    xy_points_out = np.array(xy_points_out)

    # END
    return xy_points_out


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol * 1.0

    # START
    minSetNum = 4
    maxCount = 0
    maxMatches = 0
    for k in range(num_iter):
        matches = []
        # pick random 4 matches
        for _ in range(minSetNum):
            match = random.randint(0, len(xy_src) - 1)
            matches.append(match)
        h, _ = cv2.findHomography(xy_src[matches], xy_ref[matches])
        xy_proj = KeypointProjection(xy_src, h)

        inlierCount = 0
        for i in range(xy_proj.shape[0]):
            dist = np.linalg.norm(xy_proj[i] - xy_ref[i])
            if dist < tol:
                inlierCount += 1
        if inlierCount > maxCount:
            maxCount = inlierCount
            maxMatches = matches
    # ----------------------------------
    h, _ = cv2.findHomography(xy_src[maxMatches], xy_ref[maxMatches])
    xy_proj = KeypointProjection(xy_src, h)
    inliers = []
    for i in range(xy_proj.shape[0]):
        dist = np.linalg.norm(xy_proj[i] - xy_ref[i])
        if dist < tol:
            inliers.append(i)

    h, _ = cv2.findHomography(xy_src[inliers], xy_ref[inliers])
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h
