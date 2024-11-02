import pdb
import glob
import cv2
import os
import numpy as np
import random

class PanaromaStitcher():
    def __init__(self):
        # Initialize SIFT for feature detection and matching
        self.feature_detector = cv2.SIFT_create()
        
        # FLANN parameters setup for fast matching
        TREE_INDEX = 1
        params_index = dict(algorithm=TREE_INDEX, trees=5)
        params_search = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(params_index, params_search)

    def find_features_and_match(self, image1, image2):
        # Convert images to grayscale
        grayscale_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        grayscale_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints and compute descriptors
        keypoints_1, descriptors_1 = self.feature_detector.detectAndCompute(grayscale_1, None)
        keypoints_2, descriptors_2 = self.feature_detector.detectAndCompute(grayscale_2, None)
        
        # Perform feature matching with FLANN
        matches = self.matcher.knnMatch(descriptors_1, descriptors_2, k=2)
        
        # Filter good matches using Lowe's ratio test
        verified_matches = []
        for match_1, match_2 in matches:
            if match_1.distance < 0.7 * match_2.distance:
                verified_matches.append(match_1)
                
        return keypoints_1, keypoints_2, verified_matches

    def compute_homography(self, point_pairs):
        matrix_a = []
        for point in point_pairs:
            x, y = point[0], point[1]
            X, Y = point[2], point[3]
            matrix_a.append([x, y, 1, 0, 0, 0, -X * x, -X * y, -X])
            matrix_a.append([0, 0, 0, x, y, 1, -Y * x, -Y * y, -Y])

        matrix_a = np.array(matrix_a)
        _, _, vh = np.linalg.svd(matrix_a)
        homography_matrix = (vh[-1, :].reshape(3, 3))
        homography_matrix /= homography_matrix[2, 2]
        return homography_matrix
    
    def estimate_homography_with_ransac(self, point_set, max_iterations=1000):
        optimal_inliers = []
        optimal_homography = None
        threshold = 5
        for _ in range(max_iterations):
            random_points = random.choices(point_set, k=4)
            H = self.compute_homography(random_points)
            current_inliers = []
            for pt in point_set:
                point_src = np.array([pt[0], pt[1], 1]).reshape(3, 1)
                point_dst = np.array([pt[2], pt[3], 1]).reshape(3, 1)
                estimated_dst = np.dot(H, point_src)
                estimated_dst /= estimated_dst[2]
                distance = np.linalg.norm(point_dst - estimated_dst)

                if distance < threshold:
                    current_inliers.append(pt)

            if len(current_inliers) > len(optimal_inliers):
                optimal_inliers = current_inliers
                optimal_homography = H
        
        return optimal_homography

    def calculate_homography(self, kps1, kps2, matches):
        if len(matches) < 4:
            return None
            
        # Extract corresponding points from keypoints and matches
        correspondences = []
        for match in matches:
            pt1 = kps1[match.queryIdx].pt
            pt2 = kps2[match.trainIdx].pt
            correspondences.append([pt1[0], pt1[1], pt2[0], pt2[1]])
            
        # Compute homography using RANSAC
        H_matrix = self.estimate_homography_with_ransac(correspondences)
        
        return H_matrix

    def transform_and_stitch(self, base_img, next_img, homography_matrix):
        # Determine image dimensions
        height1, width1 = base_img.shape[:2]
        height2, width2 = next_img.shape[:2]
        
        # Define corner points for base image
        corners_1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
        corners_2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)
        
        # Apply homography to base image corners
        transformed_corners_1 = cv2.perspectiveTransform(corners_1, homography_matrix)
        all_corners = np.concatenate((corners_2, transformed_corners_1), axis=0)
        
        # Calculate size for the output image
        [min_x, min_y] = np.int32(all_corners.min(axis=0).ravel())
        [max_x, max_y] = np.int32(all_corners.max(axis=0).ravel())
        
        # Define translation matrix
        translate_distance = [-min_x, -min_y]
        translation_matrix = np.array([[1, 0, translate_distance[0]], 
                                       [0, 1, translate_distance[1]], 
                                       [0, 0, 1]])
        
        # Warp base image
        output_img = cv2.warpPerspective(base_img, translation_matrix.dot(homography_matrix),
                                         (max_x - min_x, max_y - min_y))
        
        # Overlay the next image onto the output image
        output_img[translate_distance[1]:height2+translate_distance[1],
                   translate_distance[0]:width2+translate_distance[0]] = next_img
                  
        return output_img

    def make_panaroma_for_images_in(self, image_folder_path):
        image_folder = image_folder_path
        images = sorted(glob.glob(image_folder + os.sep + '*'))
        print(f'Found {len(images)} images for stitching.')
        
        if len(images) < 2:
            raise ValueError("A minimum of 2 images is required to create a panorama.")
            
        # Load the first image as the base
        stitched_image = cv2.imread(images[0])
        homography_matrices = []
        
        # Iterate through the images
        for i in range(1, len(images)):
            # Load the subsequent image
            current_image = cv2.imread(images[i])
            
            # Find features and matches
            kps1, kps2, good_matches = self.find_features_and_match(stitched_image, current_image)
            
            # Calculate homography
            homography = self.calculate_homography(kps1, kps2, good_matches)
            if homography is None:
                print(f"Warning: No suitable homography found for image {i}")
                continue
                
            homography_matrices.append(homography)
            
            # Warp and merge images
            try:
                stitched_image = self.transform_and_stitch(stitched_image, current_image, homography)
            except cv2.error as error:
                print(f"Error in warping image {i}: {error}")
                continue
        
        # Crop the final panorama to remove empty edges
        gray_img = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, width, height = cv2.boundingRect(contours[0])
        stitched_image = stitched_image[y:y+height, x:x+width]
        
        return stitched_image, homography_matrices
