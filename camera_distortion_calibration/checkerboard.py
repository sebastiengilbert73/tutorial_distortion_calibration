import copy
import os
import logging
import argparse
import cv2
import numpy as np
import utilities.blob_analysis as blob_analysis

class CheckerboardIntersections():
    def __init__(self,
                 adaptive_threshold_block_side=17,
                 adaptive_threshold_bias=-10,
                 correlation_threshold=0.8,
                 debug_directory=None
                 ):
        self.adaptive_threshold_block_side = adaptive_threshold_block_side
        if self.adaptive_threshold_block_side % 2 == 0:
            self.adaptive_threshold_block_side += 1
        self.adaptive_threshold_bias = adaptive_threshold_bias
        self.correlation_threshold = correlation_threshold
        self.debug_directory = debug_directory
        if self.debug_directory is not None:
            if not os.path.exists(self.debug_directory):
                os.makedirs(self.debug_directory)

    def FindIntersections(self, checkerboard_img):
        annotated_img = copy.deepcopy(checkerboard_img)

        # Convert to binary with adaptive threshold
        grayscale_pattern_img = cv2.cvtColor(checkerboard_img, cv2.COLOR_BGR2GRAY)
        thresholded_pattern_img = cv2.adaptiveThreshold(grayscale_pattern_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                        cv2.THRESH_BINARY, self.adaptive_threshold_block_side,
                                                        self.adaptive_threshold_bias)

        intersection1, intersection2 = self.IntersectionPatterns()
        # Correlate with the intersection patterns
        shifted_correlation1_img = cv2.matchTemplate(thresholded_pattern_img, intersection1, cv2.TM_CCOEFF_NORMED)
        shifted_correlation2_img = cv2.matchTemplate(thresholded_pattern_img, intersection2, cv2.TM_CCOEFF_NORMED)
        # Copy the correlation images in images with the size of the original image
        correlation1_img = np.zeros((checkerboard_img.shape[0], checkerboard_img.shape[1]), dtype=float)
        correlation2_img = np.zeros((checkerboard_img.shape[0], checkerboard_img.shape[1]), dtype=float)
        correlation1_img[intersection1.shape[0] // 2: intersection1.shape[0] // 2 + shifted_correlation1_img.shape[0],
        intersection1.shape[1] // 2: intersection1.shape[1] // 2 + shifted_correlation1_img.shape[
            1]] = shifted_correlation1_img
        correlation2_img[intersection2.shape[0] // 2: intersection2.shape[0] // 2 + shifted_correlation2_img.shape[0],
        intersection2.shape[1] // 2: intersection2.shape[1] // 2 + shifted_correlation2_img.shape[
            1]] = shifted_correlation2_img

        # Threshold the correlation images
        _, thresholded_correlation1_img = cv2.threshold(correlation1_img, self.correlation_threshold, 255, cv2.THRESH_BINARY)
        _, thresholded_correlation2_img = cv2.threshold(correlation2_img, self.correlation_threshold, 255, cv2.THRESH_BINARY)
        thresholded_correlation1_img = thresholded_correlation1_img.astype(np.uint8)
        thresholded_correlation2_img = thresholded_correlation2_img.astype(np.uint8)

        # Blob analysis of the thresholded correlation images
        blob_detector = blob_analysis.BinaryBlobDetector()
        seedPoint_boundingBox1_list, blobs_annotated1_img = blob_detector.DetectBlobs(
            thresholded_correlation1_img.astype(np.uint8))
        seedPoint_boundingBox2_list, blobs_annotated2_img = blob_detector.DetectBlobs(
            thresholded_correlation2_img.astype(np.uint8))

        # Find the center of mass of each blob
        center_of_mass1_list = self.CentersOfMass(seedPoint_boundingBox1_list, thresholded_correlation1_img)
        for (x, y) in center_of_mass1_list:
            cv2.circle(annotated_img, (round(x), round(y)), 3, (255, 0, 0), thickness=1)
        center_of_mass2_list = self.CentersOfMass(seedPoint_boundingBox2_list, thresholded_correlation2_img)
        for (x, y) in center_of_mass2_list:
            cv2.circle(annotated_img, (round(x), round(y)), 3, (0, 255, 0), thickness=1)
        intersections_list = center_of_mass1_list + center_of_mass2_list

        if self.debug_directory is not None:
            annotated_img_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_annotated.png")
            cv2.imwrite(annotated_img_filepath, annotated_img)
            thresholded_pattern_img_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_thresholdedPattern.png")
            cv2.imwrite(thresholded_pattern_img_filepath, thresholded_pattern_img)
            intersection1_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_intersection1.png")
            cv2.imwrite(intersection1_filepath, intersection1)
            intersection2_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_intersection2.png")
            cv2.imwrite(intersection2_filepath, intersection2)
            correlation1_img_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_correlation1.png")
            cv2.imwrite(correlation1_img_filepath, 127 + 127 * correlation1_img)
            correlation2_img_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_correlation2.png")
            cv2.imwrite(correlation2_img_filepath, 127 + 127 * correlation2_img)
            thresholded_correlation1_img_filepath = os.path.join(self.debug_directory,
                                                                 "CheckerboardIntersections_FindIntersections_thresholdedCorrelation1.png")
            cv2.imwrite(thresholded_correlation1_img_filepath, thresholded_correlation1_img)
            thresholded_correlation2_img_filepath = os.path.join(self.debug_directory,
                                                                 "CheckerboardIntersections_FindIntersections_thresholdedCorrelation2.png")
            cv2.imwrite(thresholded_correlation2_img_filepath, thresholded_correlation2_img)
            blobs_annotated1_img_filepath = os.path.join(self.debug_directory, "CheckerboardIntersections_FindIntersections_blobsAnnotated1.png")
            cv2.imwrite(blobs_annotated1_img_filepath, blobs_annotated1_img)

        return intersections_list

    def IntersectionPatterns(self):
        intersection1 = np.zeros((self.adaptive_threshold_block_side, self.adaptive_threshold_block_side), dtype=np.uint8)
        intersection1[0: self.adaptive_threshold_block_side//2, 0: self.adaptive_threshold_block_side//2] = 255
        intersection1[self.adaptive_threshold_block_side//2 + 1:, self.adaptive_threshold_block_side//2 + 1:] = 255
        intersection2 = 255 - intersection1
        intersection1[self.adaptive_threshold_block_side//2, :] = 127
        intersection1[:, self.adaptive_threshold_block_side//2] = 127
        intersection2[self.adaptive_threshold_block_side // 2, :] = 127
        intersection2[:, self.adaptive_threshold_block_side // 2] = 127
        return intersection1, intersection2

    def CentersOfMass(self, seedPoint_boundingBox_list, binary_img):
        center_of_mass_list = []
        for seed_point, bounding_box in seedPoint_boundingBox_list:
            points_list = blob_analysis.PointsOfBlob(binary_img, seed_point, bounding_box)
            center_of_mass = blob_analysis.CenterOfMass(points_list)
            center_of_mass_list.append(center_of_mass)
        return center_of_mass_list