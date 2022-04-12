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
            print("Y", np.abs(tScaling - tScalingComp), np.abs(tRotation - tRotationComp))
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
    # 1. matched_pairs 순회
    # 2. 각 match에서 transformation 구하기 (translation, scaling, rotation)
    # 3. 다른 모든 match를 다시 순회하면서 inlier 개수 세기
    # 4. 순회가 끝나면 inlier가 가장 많은 match 선택
    # 5. 해당 match에서 inlier만 추출

    maxCount = 0
    maxIndex = 0
    for i, (index1, index2) in enumerate(matched_pairs):
        tScaling, tRotation = getTranform(keypoints1[index1], keypoints2[index2])
        inlierCount = 0
        for comp1, comp2 in matched_pairs:
            if(comp1 == index1 and comp2 == index2):
                continue
            tScalingComp, tRotationComp = getTranform(keypoints1[comp1], keypoints2[comp2])
            if(checkRange(tScaling, tRotation, tScalingComp, tRotationComp, orient_agreement, scale_agreement)):
                inlierCount += 1
        if(inlierCount > maxCount):
            maxCount = inlierCount
            maxIndex = i
    
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
    angleTable = np.arccos(np.dot(descriptors1, descriptors2.T))
    for index1, angleArray in enumerate(angleTable):
        sortedAngleArray = list(enumerate(angleArray))
        sortedAngleArray.sort(key=lambda t:t[1])
        matchSt = sortedAngleArray[0]
        matchNd = sortedAngleArray[1]
        ratio = matchSt[1] / matchNd[1]
        if(ratio < threshold):
            matched_pairs.append([index1, matchNd[0]])
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
