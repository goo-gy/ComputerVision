from matplotlib import scale
import numpy as np
import math
import random


def getTranform(feat1, feat2):
    tScaling = feat1[2] / feat2[2]
    tRotation = feat1[3] - feat2[3]
    tRotation = (tRotation + 2 * np.pi) % (2 * np.pi)  # delete negative
    tRotation = tRotation * 180 / np.pi                # radian to degree
    return tScaling, tRotation

def checkRange(tScaling, tRotation, tScalingComp, tRotationComp, orient_agreement, scale_agreement):
    tScaleRange = tScaling * scale_agreement
    if(np.abs(tScaling - tScalingComp) < tScaleRange):
        if(np.abs(tRotation - tRotationComp) < orient_agreement or np.abs(tRotation - tRotationComp) > 360 - orient_agreement):
            return True
    return False

def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """ 
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    maxCount = 0
    maxIndex = 0
    for k in range(10):
        # matched_pairs에서 랜덤으로 match 선택
        i = random.randint(0, len(matched_pairs) - 1)
        (index1, index2) = matched_pairs[i]
        # match에서 transformation을 구한다 (scaling, rotation)
        tScaling, tRotation = getTranform(keypoints1[index1], keypoints2[index2])
        inlierCount = 0
        # 다른 모든 match를 다시 순회하면서 inlier counting
        for comp1, comp2 in matched_pairs:
            if(comp1 == index1 and comp2 == index2):
                continue
            tScalingComp, tRotationComp = getTranform(keypoints1[comp1], keypoints2[comp2])
            if(checkRange(tScaling, tRotation, tScalingComp, tRotationComp, orient_agreement, scale_agreement)):
                inlierCount += 1
        # inlier가 가장 많은 match 선택
        if(inlierCount > maxCount):
            maxCount = inlierCount
            maxIndex = i
    
    # 선택된 match에서 동일한 방식으로 inlier만 추출하여 return
    index1, index2 = matched_pairs[maxIndex]
    largest_set = [matched_pairs[maxIndex]]
    tScaling, tRotation = getTranform(keypoints1[index1], keypoints2[index2])
    inlierCount = 0
    for comp1, comp2 in matched_pairs:
        if(comp1 == index1 and comp2 == index2):
            continue
        tScalingComp, tRotationComp = getTranform(keypoints1[comp1], keypoints2[comp2])
        if(checkRange(tScaling, tRotation, tScalingComp, tRotationComp, orient_agreement, scale_agreement)):
            largest_set.append([comp1, comp2])
    
    ## END
    assert isinstance(largest_set, list)
    return largest_set



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
    ## START
    ## the following is just a placeholder to show you the output format
    matched_pairs = []
    # dot 연산을 통해 descriptor1의 각 벡터와 descriptor 2의 벡터에 대해서 모두 내적한 행렬을 생성한다.
    # 모든 element에 arc cosine를 적용해 angle을 생성한다.
    angleTable = np.arccos(np.dot(descriptors1, descriptors2.T))
    for index1, angleArray in enumerate(angleTable):
        # vector1과 angle이 가장 작은 vector2를 찾는다.
        sortedAngleArray = list(enumerate(angleArray))
        sortedAngleArray.sort(key=lambda t:t[1])
        matchSt = sortedAngleArray[0]
        matchNd = sortedAngleArray[1]
        # best match와 second match의 angle ratio가 threshold보다 작을 때만 선택한다.
        ratio = np.abs(matchSt[1] / matchNd[1])
        if(ratio <= threshold):
            matched_pairs.append([index1, matchSt[0]])
    ## END
    return matched_pairs


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
