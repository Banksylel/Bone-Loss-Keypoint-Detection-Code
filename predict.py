

import argparse
from Tools.Pred import Pred_Model
import os
import ast


def str2bool(value):
    """Convert string to boolean for command line arguments."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1'):
        return True
    elif value.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def main():
    """
    WRAPPER SCRIPT, to evaluate YOLOv8-pose using 5-fold cross-validation. Allows calling Eval.py from command line, notebook, or other scripts. 

    Prints evaluations and saves to csv files.
    """
    parser = argparse.ArgumentParser(description="Evaluate Model Wrapper")
    
    # Arguments
    parser.add_argument('--code-directory', type=str, required=True, help='Path to the code directory')
    parser.add_argument('--dataset-dir-standard', type=str, required=True, help='Path to the rotating bounding box dataset')
    parser.add_argument('--weight-file', type=str, required=True, help='full path to weight file')
    parser.add_argument('--seg-weights', type=str, required=True, help='full path to seg weight file')
    parser.add_argument('--ignore-box-classes', type=str, default=None, help='string of list shape, containing box class number to ignore from plotting')
    parser.add_argument('--save-loc', type=str, required=True, help='full path to save location for results')
    parser.add_argument('--post-process-kpts', type=str2bool, default=False, help='Flag to include keypoint post-processing')
    parser.add_argument('--image-size', type=int, default=640, help='Image size for evaluation')
    parser.add_argument('--furcation-dist-thresh', type=float, default=0.05, help='Threshold for furcation distance')
    parser.add_argument('--box-iou-thresh', type=float, default=0.35, help='IOU threshold to link predicted and target boxes')
    parser.add_argument('--non-max-merge-thresh', type=float, default=0.5, help='Non-max suppression threshold for merging boxes')
    parser.add_argument('--pred-seg-iou', type=float, default=0.7, help='IOU threshold for predicted segmentation masks')
    parser.add_argument('--pred-seg-conf', type=float, default=0.3, help='Confidence threshold for predicted segmentation masks')
    parser.add_argument('--pred-kpts-iou', type=float, default=0.7, help='IOU threshold for predicted keypoints')
    parser.add_argument('--pred-kpts-conf', type=float, default=0.3, help='Confidence threshold for predicted keypoints')

                        
    args = parser.parse_args()

    # creates save location if does not exist
    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)
    # changes save loc subfolder name if already exists
    save_loc_temp = os.path.join(args.save_loc, 'run_1')
    if os.path.exists(save_loc_temp):
        i = 1
        while os.path.exists(os.path.join(args.save_loc, f'run_{i}')):
            i += 1
        save_loc_temp = os.path.join(args.save_loc, f'run_{i}')
    os.makedirs(save_loc_temp)
    args.save_loc = save_loc_temp

    best_seg_weights = args.seg_weights
    weight_path_orig = args.weight_file
    
    
    pred_imgs_orig = args.dataset_dir_standard

    # Keypoint classes
    kpt_classes = [
        'CEJ-m', 'BL-m', 'RL-m', 'CEJ-d', 'BL-d', 'RL-d', 'RL-c', 
        'FA', 'FBL-m', 'FBL-d', 'ARR'
    ]
    box_classes = [
        'single_root', 'double_root', 'triple_root', 'ARR', 'PLS'
    ]
    # ignore_box_classes = [
    #     0, 1, 2
    # ]
    if args.ignore_box_classes is not None:
        ignore_box_classes = ast.literal_eval(args.ignore_box_classes)
    else:
        ignore_box_classes = None

    # Evaluate model
    Pred_Model(
        weight_path_orig, pred_imgs_orig, best_seg_weights, ignore_box_classes, box_classes, args
    )

if __name__ == "__main__":
    main()

