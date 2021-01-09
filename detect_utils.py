import cv2
from scipy.spatial import distance as dist


def mouth_aspect_ratio(mouth) -> float:
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6])  # 49, 55
    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)
    return mar


def resize_img(frame_crop, max_width, max_height):
    height, width = frame_crop.shape[:2]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        # resize image
        frame_crop = cv2.resize(frame_crop, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame_crop
