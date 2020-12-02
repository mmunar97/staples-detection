from staples_detection.assets.example_assets import get_example_asset
from staples_detection.base.staple_detection_methods import StapleDetectionMethod
from staples_detection.staple_detector import StapleDetector

from skimage.io import imsave

if __name__ == "__main__":

    detector = StapleDetector(get_example_asset(number=1))

    horizontal_staple_detection_result = detector.detect_staples(StapleDetectionMethod.HORIZONTAL_GRADIENT)
    vertical_staple_detection_result = detector.detect_staples(StapleDetectionMethod.VERTICAL_GRADIENT)
    combined_staple_detection_result = detector.detect_staples(StapleDetectionMethod.COMBINED_GRADIENT)

    print(f"Time spent – Horizontal gradient: {horizontal_staple_detection_result.elapsed_time} s")
    print(f"Time spent – Vertical gradient: {vertical_staple_detection_result.elapsed_time} s")
    print(f"Time spent – Combined gradient: {combined_staple_detection_result.elapsed_time} s")


    #imsave("/Users/marcmunar/Desktop/horizontal_result001.png", horizontal_staple_detection_result.colormask)
    #imsave("/Users/marcmunar/Desktop/vertical_result001.png", vertical_staple_detection_result.colormask)
    #imsave("/Users/marcmunar/Desktop/combined_result001.png", combined_staple_detection_result.colormask)
