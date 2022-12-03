import copy
import os
import logging
import cv2
from camera_distortion_calibration import checkerboard
import camera_distortion_calibration.radial_distortion as radial_dist
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import pickle

logging.basicConfig(level=logging.DEBUG, format='%(asctime)-15s %(levelname)s \t%(message)s')

def main():
    logging.debug("calibrate_with_checkerboard.main()")

    output_directory = "./output"
    checkerboard_image_filepath = "./images/cam_left.png"
    adaptive_threshold_block_side = 17
    adaptive_threshold_bias = -10
    correlation_threshold = 0.8
    grid_shapeHW = (6, 6)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the checkerboard image
    checkerboard_img = cv2.imread(checkerboard_image_filepath)
    annotated_img = copy.deepcopy(checkerboard_img)
    # Display the checkerboard
    cv2.imshow("Checkerboard", checkerboard_img)
    cv2.waitKey(0)

    # Find the checkerboard intersections, which will be our feature points that belong to a plane
    checkerboard_intersections = checkerboard.CheckerboardIntersections(
        adaptive_threshold_block_side=adaptive_threshold_block_side,
        adaptive_threshold_bias=adaptive_threshold_bias,
        correlation_threshold=correlation_threshold,
        debug_directory=output_directory
    )
    intersections_list = checkerboard_intersections.FindIntersections(checkerboard_img)
    # Display the found intersections
    intersections_annotated_img = cv2.imread(os.path.join(output_directory, "CheckerboardIntersections_FindIntersections_annotated.png"))
    cv2.imshow("Found feature points", intersections_annotated_img)
    cv2.waitKey(0)

    # Create a RadialDistortion object, that will optimize its parameters
    radial_distortion = radial_dist.RadialDistortion((checkerboard_img.shape[0], checkerboard_img.shape[1]))
    # Start the optimization
    epoch_loss_center_alpha_list = radial_distortion.Optimize(intersections_list, grid_shapeHW)
    Plot([epoch for epoch, _, _, _ in epoch_loss_center_alpha_list],
         [[loss for _, loss, _, _ in epoch_loss_center_alpha_list]], ["loss"])

    # Undistort the points
    for p in intersections_list:
        undistorted_p = radial_distortion.UndistortPoint(p)
        cv2.circle(annotated_img, (round(undistorted_p[0]), round(undistorted_p[1])), 3, (255, 0, 0), thickness=-1)
    cv2.imshow("The undistorted feature points", annotated_img)
    cv2.waitKey()

    # Undistort the checkerboard image
    undistorted_checkerboard_img = radial_distortion.UndistortImage(checkerboard_img)
    cv2.imshow("The undistorted checkerboard image (no filling)", undistorted_checkerboard_img)
    cv2.waitKey()
    cv2.imwrite(os.path.join(output_directory, "calibrateWithCheckerboard_main_undistortedCheckerboard.png"), undistorted_checkerboard_img)

    # Undistort the checkerboard image with filling
    undistorted_checkerboard_filled_img = radial_distortion.UndistortImage(checkerboard_img, fill_with_local_median=True)
    cv2.imshow("The undistorted checkerboard image (with filling)", undistorted_checkerboard_filled_img)
    cv2.waitKey()
    cv2.imwrite(os.path.join(output_directory, "calibrateWithCheckerboard_main_undistortedCheckerboardFillled.png"),
                undistorted_checkerboard_filled_img)

    # Save the calibration
    # It will be possible to load the calibration file with the following code:
    # with open(filepath, 'rb') as obj_file:
    #    radial_distortion_obj = pickle.load(obj_file)
    calibration_filepath = os.path.join(output_directory, "calibration.pkl")
    with open(calibration_filepath, 'wb') as calibration_file:
        pickle.dump(radial_distortion, calibration_file, pickle.HIGHEST_PROTOCOL)
    logging.info(f"Saved calibration file to {calibration_filepath}")

def Plot(xs, ys_list, y_labels_list):
    fig, ax = plt.subplots()
    for plot_ndx in range(len(ys_list)):
        ax.plot(xs, ys_list[plot_ndx])
        ax.set(xlabel='epoch', ylabel=y_labels_list[plot_ndx])
    ax.legend()
    ax.grid()
    plt.show()

if __name__ == '__main__':
    main()