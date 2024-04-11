import argparse
import cv2
import numpy as np


# argument parser
def getArgs():
    parser = argparse.ArgumentParser(
        description='''Demo script showing various image alignment methods
                    including, phase correlation, feature based matching
                    and whole image based optimization.''',
        epilog='''post bug reports to the github repository''')

    parser.add_argument('-im1',
                        '--image_1',
                        help='image to reference',
                        required=True)

    parser.add_argument('-im2',
                        '--image_2',
                        help='image to match',
                        required=True)

    parser.add_argument('-m',
                        '--mode',
                        help='registation mode: translation, ecc or feature',
                        default='feature')

    parser.add_argument('-mf',
                        '--max_features',
                        help='maximum number of features to consider',
                        default=10000)

    parser.add_argument('-fr',
                        '--feature_retention',
                        help='fraction of features to retain',
                        default=0.15)

    parser.add_argument('-i',
                        '--iterations',
                        help='number of ecc iterations',
                        default=10000)

    parser.add_argument('-te',
                        '--termination_eps',
                        help='ecc termination value',
                        default=1e-8)

    return parser.parse_args()


# (ORB) feature based alignment
def featureAlign(im1, im2):
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    sift = cv2.SIFT_create(max_features)
    keypoints1, descriptors1 = sift.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(im2Gray, None)

    descriptors1 = descriptors1.astype(np.float32)
    descriptors2 = descriptors2.astype(np.float32)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE) #hamming
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * feature_retention)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    # cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    return im1Reg, h


if __name__ == '__main__':

    # parse arguments
    args = getArgs()

    # defaults feature values
    max_features = args.max_features
    feature_retention = args.feature_retention
    # Specify the ECC number of iterations.
    number_of_iterations = args.iterations

    # Specify the ECC threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = args.termination_eps

    # Read the images to be aligned
    im1 = cv2.imread(args.image_1);
    im2 = cv2.imread(args.image_2);

    # Switch between alignment modes
    if args.mode == "feature":
        # align and write to disk
        aligned, warp_matrix = featureAlign(im1, im2)
        cv2.imwrite("reg_image.jpg",
                    aligned,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(warp_matrix)
    elif args.mode == "ecc":
        aligned, warp_matrix = eccAlign(im1, im2)
        cv2.imwrite("reg_image.jpg",
                    aligned,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(warp_matrix)
    elif args.mode == "rotation":
        rotated, rotationMatrix = rotationAlign(im1, im2)
        cv2.imwrite("reg_image.jpg",
                    rotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(rotationMatrix)
    else:
        warp_matrix = translation(im1, im2)
