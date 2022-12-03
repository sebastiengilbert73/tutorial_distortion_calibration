import cv2
import copy


class BinaryBlobDetector():
    def __init__(self, maximum_center_offset=31):
        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByColor = False
        self.params.filterByArea = False
        self.params.filterByCircularity = False
        self.params.filterByConvexity = False
        self.params.filterByInertia = False
        self.params.minArea = 0
        self.params.maxArea = float('inf')
        self.params.minCircularity = 0
        self.params.maxCircularity = 1
        self.params.minConvexity = 0
        self.params.maxConvexity = 1
        self.params.minInertiaRatio = 0
        self.params.maxInertiaRatio = 1
        # https://stackoverflow.com/questions/59217213/opencv-blob-detector-has-offset-on-found-position
        self.params.minThreshold = 128
        self.params.maxThreshold = 129
        self.params.thresholdStep = 1
        self.params.minRepeatability = 1
        self.maximum_center_offset = maximum_center_offset

        self.blob_detector = cv2.SimpleBlobDetector_create(self.params)

    def DetectBlobs(self, binary_img):
        keypoints = self.blob_detector.detect(
            255 - binary_img)  # cv2.SimpleBlobDetector expects black blobs on a white background

        annotated_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        img_shape_HW = binary_img.shape

        seedPoint_boundingBox_list = []
        runningNdx = 0
        for kp in keypoints:
            center = (int(round(kp.pt[0])), int(round(kp.pt[1])) )
            fill_color = ((runningNdx * 17) % 256, (runningNdx * 43) % 256, (runningNdx * 57) % 256)
            if fill_color[0] == 0 and fill_color[1] == 0 and fill_color[2] == 0:
                fill_color = (128, 128, 128)
            closest_white_point = self.ClosestWhitePoint(binary_img, center)
            if closest_white_point is not None:
                retval, _, _, bounding_box = cv2.floodFill(annotated_img, mask=None, seedPoint=closest_white_point,
                                                           newVal=fill_color)
                cv2.circle(annotated_img, closest_white_point, 5, (fill_color[0] / 2, fill_color[1] / 2, fill_color[2] / 2),
                           thickness=-1)
                runningNdx += 1
                if bounding_box[2] < img_shape_HW[1] or bounding_box[3] < img_shape_HW[0]:
                    seedPoint_boundingBox_list.append( (closest_white_point, bounding_box) )
        seedPoint_boundingBox_list = self.RemoveDuplicates(seedPoint_boundingBox_list, annotated_img)
        return seedPoint_boundingBox_list, annotated_img

    def ClosestWhitePoint(self, binary_img, point):
        if binary_img[point[1], point[0]] == 255:
            return point
        img_shape_HW = binary_img.shape
        closest_point = None
        for neighborhood_size in range(3, 2 * self.maximum_center_offset + 1, 2):
            neighborhood = [point[0] - neighborhood_size // 2, point[1] - neighborhood_size // 2,
                            neighborhood_size, neighborhood_size]
            neighborhood[0] = max(neighborhood[0], 0)
            neighborhood[1] = max(neighborhood[1], 0)
            neighborhood[2] = min(neighborhood_size, img_shape_HW[1] - neighborhood[0])
            neighborhood[3] = min(neighborhood_size, img_shape_HW[0] - neighborhood[1])
            neighborhood_img = binary_img[neighborhood[1]: neighborhood[1] + neighborhood[3],
                               neighborhood[0]: neighborhood[0] + neighborhood[2]]
            number_of_nonzero_pixels = cv2.countNonZero(neighborhood_img)
            if number_of_nonzero_pixels > 0:
                for y in range(neighborhood_img.shape[0]):
                    for x in range(neighborhood_img.shape[1]):
                        if neighborhood_img[y, x] == 255:
                            closest_point = (neighborhood[0] + x, neighborhood[1] + y)
                break
        return closest_point

    def RemoveDuplicates(self, seedPoint_boundingBox_list, annotated_img):
        cleaned_seedPoint_boundingBox_list = []
        for candidate_seedPoint_boundingBox in seedPoint_boundingBox_list:
            candidate_boundingBox_is_already_present = False
            for already_counted_pair in cleaned_seedPoint_boundingBox_list:
                if already_counted_pair[1] == candidate_seedPoint_boundingBox[1]:  # The bounding boxes coincide
                    candidate_boundingBox_is_already_present = True
                    cv2.circle(annotated_img, candidate_seedPoint_boundingBox[0], 9,
                               (0, 0, 255),
                               thickness=1)
                    break
            if not candidate_boundingBox_is_already_present:
                cleaned_seedPoint_boundingBox_list.append(candidate_seedPoint_boundingBox)
        return cleaned_seedPoint_boundingBox_list

def CenterOfMass(points_list):
    if len(points_list) == 0:
        raise ValueError(f"blob_analysis.CenterOfMass(): The list of points is empty")
    sum_x = 0.0
    sum_y = 0.0
    for (x, y) in points_list:
        sum_x += x
        sum_y += y
    center_of_mass = (sum_x/len(points_list), sum_y/len(points_list))
    return center_of_mass

def PointsOfBlob(binary_img, seed_point, bounding_box=None):
    floodfilled_img = copy.deepcopy(binary_img)
    # Floodfill with zero, starting from the seed point
    _, floodfilled_img, _, _ = cv2.floodFill(floodfilled_img, None, seed_point, 0)
    blob_img = binary_img - floodfilled_img  # Only the blob remains
    points_list = []
    range_y = [0, blob_img.shape[0]]
    range_x = [0, blob_img.shape[1]]
    if bounding_box is not None:
        range_y = [bounding_box[1], bounding_box[1] + bounding_box[3]]
        range_x = [bounding_box[0], bounding_box[0] + bounding_box[2]]
    for y in range(range_y[0], range_y[1]):
        for x in range(range_x[0], range_x[1]):
            if blob_img[y, x] > 0:
                points_list.append((x, y))
    return points_list
