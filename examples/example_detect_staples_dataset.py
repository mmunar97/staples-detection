from staples_detection.assets.example_assets import get_example_asset
from staples_detection.base.staple_detection_methods import StapleDetectionMethod
from staples_detection.staple_detector import StapleDetector

from skimage.io import imsave

import cv2
import os
import pandas
from tqdm import tqdm

if __name__ == "__main__":

    r"""
    detector = StapleDetector(get_example_asset(number=1))

    horizontal_staple_detection_result = detector.detect_staples(StapleDetectionMethod.HORIZONTAL_GRADIENT)
    vertical_staple_detection_result = detector.detect_staples(StapleDetectionMethod.VERTICAL_GRADIENT)
    combined_staple_detection_result = detector.detect_staples(StapleDetectionMethod.COMBINED_GRADIENT)
    canny_staple_detection_result = detector.detect_staples(StapleDetectionMethod.CANNY)

    print(f"Time spent – Horizontal gradient: {horizontal_staple_detection_result.elapsed_time} s")
    print(f"Time spent – Vertical gradient: {vertical_staple_detection_result.elapsed_time} s")
    print(f"Time spent – Combined gradient: {combined_staple_detection_result.elapsed_time} s")
    print(f"Time spent – Canny: {canny_staple_detection_result.elapsed_time} s")

    imsave(r"C:\Users\Usuario\Desktop\Staples\horizontal_result001.png", horizontal_staple_detection_result.colormask)
    imsave(r"C:\Users\Usuario\Desktop\Staples\vertical_result001.png", vertical_staple_detection_result.colormask)
    imsave(r"C:\Users\Usuario\Desktop\Staples\combined_result001.png", combined_staple_detection_result.colormask)
    imsave(r"C:\Users\Usuario\Desktop\Staples\canny_result001.png", canny_staple_detection_result.colormask)
    """

    DATASET_PATH = r"C:\Users\Usuario\OneDrive - Universitat de les Illes Balears\UIB\Investigacio\Proyectos I+D\Redscar\Images\Dataset\SUBSETS\MACHINE_LEARNING_DATASET"
    GT_PATH = r"C:\Users\Usuario\OneDrive - Universitat de les Illes Balears\UIB\Investigacio\Proyectos I+D\Redscar\Images\Dataset\REDSCAR_DATASET\GT_STAPLES_MASK"
    SAVING_PATH = r"Z:\UIB_EXPERIMENTS\STAPLES_REMOVAL_EXPERIMENTS\RawStapleDetection"

    TEST_IMAGES = os.path.join(DATASET_PATH, "test", "IMAGES")
    TRAIN_IMAGES = os.path.join(DATASET_PATH, "train", "IMAGES")

    # Computing test results
    test_results = {"IMAGES": [],
                    "HORIZONTAL_TP": [], "HORIZONTAL_TN": [], "HORIZONTAL_FP": [], "HORIZONTAL_FN": [],
                    "VERTICAL_TP": [], "VERTICAL_TN": [], "VERTICAL_FP": [], "VERTICAL_FN": [],
                    "COMBINED_TP": [], "COMBINED_TN": [], "COMBINED_FP": [], "COMBINED_FN": [],
                    "CANNY_TP": [], "CANNY_TN": [], "CANNY_FP": [], "CANNY_FN": []}

    for image_name in tqdm(os.listdir(TEST_IMAGES)):
        image_name_no_extension = image_name.replace(".png", "")

        image = cv2.imread(os.path.join(TEST_IMAGES, image_name))
        gt = cv2.imread(os.path.join(GT_PATH, image_name), cv2.IMREAD_GRAYSCALE).astype(bool)

        detector = StapleDetector(image=image)
        horizontal_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.HORIZONTAL_GRADIENT,
                                                                     ground_truth_mask=gt)
        vertical_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.VERTICAL_GRADIENT,
                                                                   ground_truth_mask=gt)
        combined_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.COMBINED_GRADIENT,
                                                                   ground_truth_mask=gt)
        canny_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.CANNY,
                                                                ground_truth_mask=gt)

        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension+"_horizontalgradient_colormask.png"), horizontal_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_horizontalgradient_mask.png"), horizontal_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_verticalgradient_colormask.png"), vertical_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_verticalgradient_mask.png"), vertical_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_combinedgradient_colormask.png"), combined_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_combinedgradient_mask.png"), combined_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_cannygradient_colormask.png"), canny_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "test", image_name_no_extension + "_cannygradient_mask.png"), canny_staple_detection_result.final_mask)

        test_results["IMAGES"].append(image_name)
        test_results["HORIZONTAL_TP"].append(horizontal_staple_detection_result.performance.true_positive)
        test_results["HORIZONTAL_TN"].append(horizontal_staple_detection_result.performance.true_negative)
        test_results["HORIZONTAL_FP"].append(horizontal_staple_detection_result.performance.false_positive)
        test_results["HORIZONTAL_FN"].append(horizontal_staple_detection_result.performance.false_negative)

        test_results["VERTICAL_TP"].append(vertical_staple_detection_result.performance.true_positive)
        test_results["VERTICAL_TN"].append(vertical_staple_detection_result.performance.true_negative)
        test_results["VERTICAL_FP"].append(vertical_staple_detection_result.performance.false_positive)
        test_results["VERTICAL_FN"].append(vertical_staple_detection_result.performance.false_negative)

        test_results["COMBINED_TP"].append(combined_staple_detection_result.performance.true_positive)
        test_results["COMBINED_TN"].append(combined_staple_detection_result.performance.true_negative)
        test_results["COMBINED_FP"].append(combined_staple_detection_result.performance.false_positive)
        test_results["COMBINED_FN"].append(combined_staple_detection_result.performance.false_negative)

        test_results["CANNY_TP"].append(canny_staple_detection_result.performance.true_positive)
        test_results["CANNY_TN"].append(canny_staple_detection_result.performance.true_negative)
        test_results["CANNY_FP"].append(canny_staple_detection_result.performance.false_positive)
        test_results["CANNY_FN"].append(canny_staple_detection_result.performance.false_negative)

    test_results_df = pandas.DataFrame(test_results)
    test_results_df.to_csv(os.path.join(SAVING_PATH, "test.csv"))

    # Computing train results
    train_results = {"IMAGES": [],
                     "HORIZONTAL_TP": [], "HORIZONTAL_TN": [], "HORIZONTAL_FP": [], "HORIZONTAL_FN": [],
                     "VERTICAL_TP": [], "VERTICAL_TN": [], "VERTICAL_FP": [], "VERTICAL_FN": [],
                     "COMBINED_TP": [], "COMBINED_TN": [], "COMBINED_FP": [], "COMBINED_FN": [],
                     "CANNY_TP": [], "CANNY_TN": [], "CANNY_FP": [], "CANNY_FN": []}

    for image_name in tqdm(os.listdir(TRAIN_IMAGES)):
        image_name_no_extension = image_name.replace(".png", "")

        image = cv2.imread(os.path.join(TRAIN_IMAGES, image_name))
        gt = cv2.imread(os.path.join(GT_PATH, image_name), cv2.IMREAD_GRAYSCALE).astype(bool)

        detector = StapleDetector(image=image)
        horizontal_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.HORIZONTAL_GRADIENT,
                                                                     ground_truth_mask=gt)
        vertical_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.VERTICAL_GRADIENT,
                                                                   ground_truth_mask=gt)
        combined_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.COMBINED_GRADIENT,
                                                                   ground_truth_mask=gt)
        canny_staple_detection_result = detector.detect_staples(method=StapleDetectionMethod.CANNY,
                                                                ground_truth_mask=gt)

        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_horizontalgradient_colormask.png"), horizontal_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_horizontalgradient_mask.png"), horizontal_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_verticalgradient_colormask.png"), vertical_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_verticalgradient_mask.png"), vertical_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_combinedgradient_colormask.png"), combined_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_combinedgradient_mask.png"), combined_staple_detection_result.final_mask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_cannygradient_colormask.png"), canny_staple_detection_result.colormask)
        cv2.imwrite(os.path.join(SAVING_PATH, "train", image_name_no_extension + "_cannygradient_mask.png"), canny_staple_detection_result.final_mask)

        train_results["IMAGES"].append(image_name)
        train_results["HORIZONTAL_TP"].append(horizontal_staple_detection_result.performance.true_positive)
        train_results["HORIZONTAL_TN"].append(horizontal_staple_detection_result.performance.true_negative)
        train_results["HORIZONTAL_FP"].append(horizontal_staple_detection_result.performance.false_positive)
        train_results["HORIZONTAL_FN"].append(horizontal_staple_detection_result.performance.false_negative)

        train_results["VERTICAL_TP"].append(vertical_staple_detection_result.performance.true_positive)
        train_results["VERTICAL_TN"].append(vertical_staple_detection_result.performance.true_negative)
        train_results["VERTICAL_FP"].append(vertical_staple_detection_result.performance.false_positive)
        train_results["VERTICAL_FN"].append(vertical_staple_detection_result.performance.false_negative)

        train_results["COMBINED_TP"].append(combined_staple_detection_result.performance.true_positive)
        train_results["COMBINED_TN"].append(combined_staple_detection_result.performance.true_negative)
        train_results["COMBINED_FP"].append(combined_staple_detection_result.performance.false_positive)
        train_results["COMBINED_FN"].append(combined_staple_detection_result.performance.false_negative)

        train_results["CANNY_TP"].append(canny_staple_detection_result.performance.true_positive)
        train_results["CANNY_TN"].append(canny_staple_detection_result.performance.true_negative)
        train_results["CANNY_FP"].append(canny_staple_detection_result.performance.false_positive)
        train_results["CANNY_FN"].append(canny_staple_detection_result.performance.false_negative)

    train_results_df = pandas.DataFrame(train_results)
    train_results_df.to_csv(os.path.join(SAVING_PATH, "train.csv"))
