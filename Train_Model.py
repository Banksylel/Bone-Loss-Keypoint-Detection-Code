import argparse
import os
import numpy as np
import csv
from ultralytics import YOLO


def save_results_to_csv(results, output_file):
    """Save training results to a CSV file."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Mean", "Standard Deviation"])
        for metric, values in results.items():
            mean = np.mean(values)
            std_dev = np.std(values)
            writer.writerow([metric, mean, std_dev])


def main():
    """
    TRAIN SCRIPT, to train YOLOv8-pose using 5-fold cross-validation. Allows training from command line, notebook, or other scripts. 

    Trains models, saving results and weights to 'runs/pose' folder in code directory.
    """
    parser = argparse.ArgumentParser(description="Train Model Wrapper")

    # Arguments
    parser.add_argument('--code-directory', type=str, required=True, help='Path to the code directory')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--image-size', type=int, default=640, help='Image size for training')
    parser.add_argument('--output-csv', type=str, default="training_results.csv", help='Path to save the training results')

    args = parser.parse_args()

    # Paths
    cfg_path = os.path.join(args.code_directory, 'ultralytics/cfg/datasets')

    # Placeholder for metrics
    box_prec, box_recall, box_ap50, box_ap, pose_prec, pose_recall, pose_ap50, pose_ap = [], [], [], [], [], [], [], []

    # Perform K-Fold Cross Validation
    for fold in range(args.folds):
        print(f"Fold {fold}/{args.folds - 1}")

        # Load the YOLOv8 model
        model = YOLO('yolov8n-pose.pt')

        # Get the configuration file for the current fold
        cur_file_loc = os.path.join(cfg_path, f'CUSTOM_pose_f{fold}.yaml')

        # Train the model
        model.train(data=cur_file_loc, epochs=args.epochs, imgsz=args.image_size)

        # Validate the model
        results = model.val()

        # Collect metrics
        box_prec.append(results.box.p)
        box_recall.append(results.box.r)
        box_ap50.append(results.box.ap50)
        box_ap.append(results.box.ap)
        pose_prec.append(results.pose.p)
        pose_recall.append(results.pose.r)
        pose_ap50.append(results.pose.ap50)
        pose_ap.append(results.pose.ap)

    # Save results to CSV
    results = {
        "Box Precision": box_prec,
        "Box Recall": box_recall,
        "Box AP50": box_ap50,
        "Box AP": box_ap,
        "Pose Precision": pose_prec,
        "Pose Recall": pose_recall,
        "Pose AP50": pose_ap50,
        "Pose AP": pose_ap
    }
    save_results_to_csv(results, args.output_csv)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()