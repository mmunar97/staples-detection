import cv2
from staples_detection.base.staple_detection_methods import StapleDetectionMethod

from staples_detection import StapleDetector

if __name__ == "__main__":

    image = cv2.imread(r"assets\images\bavI7HSY6JwbPYkZpPJuzgSuV_infection=0_capture=1_resolution=1.png")
    gt = cv2.imread(r"assets\gt\bavI7HSY6JwbPYkZpPJuzgSuV_infection=0_capture=1_resolution=1.png", cv2.IMREAD_GRAYSCALE).astype(bool)

    detector = StapleDetector(image=image)
    horizontal_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.HORIZONTAL_GRADIENT,
                                                                 ground_truth_mask=gt)
    vertical_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.VERTICAL_GRADIENT,
                                                               ground_truth_mask=gt)
    combined_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.COMBINED_GRADIENT,
                                                               ground_truth_mask=gt)
    canny_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.CANNY,
                                                            ground_truth_mask=gt)
    discrete_morphology_result = detector.detect_staples(method=StapleDetectionMethod.DISCRETE_MORPHOLOGY,
                                                         ground_truth_mask=gt)
