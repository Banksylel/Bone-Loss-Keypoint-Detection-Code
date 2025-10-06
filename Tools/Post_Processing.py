import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO
from math import atan2
import copy

# calculates the new post processing keypoints based off predicted tooth segmentation
# model: the model used to predict the segmentation mask
# model_kpts: the model used to predict the keypoints
# source_dir: the directory of the image
# img_file: the image file name

# map of bounding box classes and their related keypoint visibility 1 = any visible keypoint, 0 = no visible keypoints
class_keypoint_visibility_map = {0: [1,1,0,1,1,0,1,0,0,0,0], 1: [1,1,1,1,1,1,0,1,1,1,0], 2: [1,1,1,1,1,1,1,1,1,1,0], 3: [0,0,0,0,0,0,0,0,0,0,1], 4: [0,0,0,0,0,0,0,0,0,0,0]}
default_left_clss, default_right_clss = [0,1,2], [3,4,5] # keypoint classes for left and right side by default



def central_moments_from_points(xs, ys):
    """
    Compute spatial moments m00, m10, m01 and central moments mu20, mu02, mu11
    from foreground point coordinates (xs, ys).
    xs, ys : 1D arrays of the same length Integer coordinates of foreground pixels. Coordinates are assumed in (row, col) convention: xs=row (j), ys=col (i).
    returns m00, mu20, mu02, mu11 spatial moments
    """
    if xs.size == 0:
        raise ValueError("Empty foreground set; cannot compute moments.")

    # Spatial moments
    m00 = float(xs.size)
    m10 = float(ys.sum())
    m01 = float(xs.sum())

    # Centroid (x̄, ȳ) ≡ (row_mean, col_mean)
    xbar = m01 / m00
    ybar = m10 / m00

    # Central moments (order ≤ 2)
    dx = xs - xbar
    dy = ys - ybar

    mu20 = float((dy**2).sum())  # variance along columns
    mu02 = float((dx**2).sum())  # variance along rows
    mu11 = float((dx * dy).sum())

    return m00, mu20, mu02, mu11, xbar, ybar


def principal_orientation_from_central(mu20, mu02, mu11):
    """
    Return principal axis orientation in radians in (-pi/2, pi/2], measured
    from +x (columns) toward +y (rows), i.e., relative to image horizontal.
    """
    return 0.5 * atan2(2.0 * mu11, (mu20 - mu02))


def normalise_deg(angle_deg):
    """
    Normalise angle to (-90, 90] degrees for a unique representation, and reformats angle for cv2 formatting 0deg is right
    """
    a = (angle_deg + 90.0) % 180.0
    # Ensure angle is in (-90, 90]
    if a > 90.0:
        a -= 180.0
    return a


def skew_angle_from_mask(mask, degrees_out = True):
    """
    Estimate the skew (principal orientation) of a SINGLE-OBJECT binary mask. Function also refines object to edge pixels
    mask : (2D ndarray) of bool or {0,1} Foreground (True/1) defines the object.
    degrees_out : (bool) If True, return angle in degrees. Otherwise radians.
    Returns angle (float) Orientation relative to the horizontal axis in (-90, 90] degrees (or radians). Positive angles rotate clockwise. 
    """
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")
    
    # only retains the largest object
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        # No objects found (background only)
        mask = np.zeros_like(mask)
    else:
        # Ignore background (label 0), find the largest component
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_label).astype(mask.dtype)

    # canny edge detector to detect edges, splits into xs and ys postivie pixel coordinates
    edges = cv2.Canny(mask.astype(np.uint8), 0, 1)
    xs, ys = np.nonzero(edges)


    # Compute central moments from chosen point set
    m00, mu20, mu02, mu11, xbar, ybar = central_moments_from_points(xs, ys)

    # Orientation from second-order central moments
    theta = principal_orientation_from_central(mu20, mu02, mu11)


    if degrees_out:
        return normalise_deg(np.degrees(theta))
    else:
        # Normalise by converting deg->norm->rad to keep consistent bounds
        return np.radians(normalise_deg(np.degrees(theta)))



def split_masks_deg(seg_instances, rotation_index):
    """
    Splits seg_instances into two masks from the centre point of the mask along the vertical axis,
    rotated by the given rotation_index (in degrees).
    Returns a tuple of (left_mask, right_mask).
    """
    # Find the nonzero (foreground) pixels
    ys, xs = np.where(seg_instances > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        # Empty mask, return two empty masks
        return np.zeros_like(seg_instances), np.zeros_like(seg_instances)

    # Compute the center of the mask
    cx = np.mean(xs)
    cy = np.mean(ys)

    # Convert rotation_index (degrees) to radians, and get the direction vector
    theta = np.deg2rad(rotation_index+90)
    dx = np.cos(theta)
    dy = np.sin(theta)

    # For each pixel, determine which side of the rotated vertical axis it is on
    # The axis passes through (cx, cy) and has direction (dx, dy)
    # For each pixel (x, y), compute the sign of the cross product:
    # (x - cx, y - cy) x (dx, dy) = (x - cx)*dy - (y - cy)*dx
    mask_shape = seg_instances.shape
    X, Y = np.meshgrid(np.arange(mask_shape[1]), np.arange(mask_shape[0]))
    side = (X - cx) * dy - (Y - cy) * dx

    # Create left and right masks
    left_mask = np.zeros_like(seg_instances)
    right_mask = np.zeros_like(seg_instances)

    # Left: side < 0, Right: side >= 0
    left_mask[(seg_instances > 0.5) & (side < 0)] = 1
    right_mask[(seg_instances > 0.5) & (side >= 0)] = 1
    

    # dilate left and right masks to fill in the gaps
    kernel = np.ones((3, 3), np.uint8)
    left_mask = cv2.dilate(left_mask.astype(np.uint8), kernel, iterations=1)
    right_mask = cv2.dilate(right_mask.astype(np.uint8), kernel, iterations=1)
    left_mask = left_mask.astype(seg_instances.dtype)
    right_mask = right_mask.astype(seg_instances.dtype)

    return left_mask, right_mask




def NMM(seg_results, iou_thresh=0.5):
    """
    Non Maximum Merging, merges segmentation masks of the same class if they have an IoU threshold of more than iou_thresh.
    the lower the iou_thresh, the more masks are merged.
    Returns a merged segmentation masks.
    """
    if len(seg_results) == 0:
        return seg_results

    new_seg_results = []
    running_count = 0

    while len(seg_results) > running_count:

        cur_seg = seg_results[running_count]
        # compares current mask to all other masks to find their iou, inline
        comparison_seg_results = seg_results[running_count+1:]

        cur_seg_iou = [np.logical_and(cur_seg > 0.5, mask > 0.5).sum() / np.logical_or(cur_seg > 0.5, mask > 0.5).sum() if np.logical_or(cur_seg > 0.5, mask > 0.5).sum() > 0 else 0 for mask in comparison_seg_results]
        # if the iou for each positional mask is greater than the threshold, merge the masks. removes matched dim from seg_results if it is merged
        
        index_adjust = 0
        for j in range(len(cur_seg_iou)):
            if cur_seg_iou[j] > iou_thresh and j != running_count:
                # Merge masks
                cur_seg = np.logical_or(seg_results[running_count + j + 1 - index_adjust] > 0.5, cur_seg > 0.5).astype(cur_seg.dtype)
                # Remove the current mask from the list
                seg_results = np.delete(seg_results, running_count + j + 1 - index_adjust, axis=0)
                index_adjust += 1

        # compares the current mask with new_seg_results
        pop_count = 0
        if len(new_seg_results) != 0:
            seg_iou_new = [np.logical_and(cur_seg > 0.5, mask > 0.5).sum() / np.logical_or(cur_seg > 0.5, mask > 0.5).sum() if np.logical_or(cur_seg > 0.5, mask > 0.5).sum() > 0 else 0 for mask in new_seg_results]
            for p in range(len(seg_iou_new)):
                if seg_iou_new[p] > iou_thresh:
                    # Merge masks
                    cur_seg = np.logical_or(new_seg_results[p-pop_count] > 0.5, cur_seg > 0.5).astype(cur_seg.dtype)
                    # Remove the current mask from the list
                    new_seg_results.pop(p - pop_count)
                    pop_count += 1
        
        new_seg_results.append(cur_seg)
        running_count += 1
    #converts list of numpy arrays to numpy array
    new_seg_results = np.stack(new_seg_results, axis=0)

    return(new_seg_results)

# model = the model for the segmentation mask
# model_kpts = the results for the keypoint prediction
# image_path = The image path location for the current image
# imgz = the image size for the model prediction
# class_ignore = the classes to ignore in the keypoint post processing

def post_process(model, model_kpts, image_path, args, class_ignore=[7, 8, 9, 10], img_dims_only=False, post_process_type=1):
    """
    Post processing loop for a single image. Saves visualisations if args.view_images is True.

    model: The YOLOv8-seg model object for tooth segmentation.
    model_kpts: Predicted keypoint model output object from the YOLOv8-kpt model.
    image_path: The image path location string for the current image.
    args: The arguments from command line input.
    class_ignore: Class numbers list of ints to ignore for post processing.
    img_dims_only: If True, only returns the image dimensions and rotation list.
    post_process_type: The type of post processing to use, 1 = Advanced, 0 = Basic (1 is used for paper).



    Returns (object_new_kpts, object_pred_kpts, object_distance, width, height, final_rotation_list), (new keypoints, original keypoints, average distance between keypoints and closest edge pixel for each object, image width, image height, rotation list for each object)
    """

    imgz=args.image_size
    show_images=args.view_images
    non_max_merge=args.non_max_merge_thresh
    iou_pred=args.pred_seg_iou
    conf_pred=args.pred_seg_conf

    if show_images:
        arrow_save_loc = os.path.join(args.save_loc, "arrow_images")
        img = image_path.split('/')[-1].split('//')[-1].split('\\')[-1]
        os.makedirs(arrow_save_loc, exist_ok=True)
    # Load the image
    image = cv2.imread(image_path)

    # predicts segmentation mask
    pred = model.predict(image_path, conf=conf_pred, iou=iou_pred, save=False, imgsz=imgz)

    if len(model_kpts) != 1:
        print("ERROR: model output contains more than one images' predictions")

    # shows the keypoints
    kpt_results = model_kpts[0].keypoints.xyn

    # extracts keypoint bounding boxes
    kpt_box_clss = model_kpts[0].boxes.cls


    seg_results = pred[0].masks.data

    # converts seg_results from tensor to numpy array
    seg_results = seg_results.cpu().numpy()


    # non maximum merging for seg_results (if IoU is greater than non_max_merge, then it combines the masks)
    if non_max_merge is not None:
        seg_results = NMM(seg_results, iou_thresh=non_max_merge)



    # converts kpt_results from tensor to numpy array
    kpt_results = kpt_results.cpu().numpy()




    # extract image size from seg_results
    width, height = seg_results.shape[2], seg_results.shape[1]
    # changes the shape of image to the evaluation size
    image = cv2.resize(image, (width, height))

    if kpt_results is None or len(kpt_results) == 0:
        print("No keypoints detected, skipping post-processing.")
        return kpt_results, kpt_results, [], width, height

    # changes the normalised keypints and boxes to the image size
    kpt_results = kpt_results * [width, height]
    

    # pkt_results to list
    original_kpts_old = kpt_results.tolist()
    original_kpts_new = kpt_results.tolist()

    object_distance = [] # the average distance between the keypoints and the closest edge pixel for each group of keypoints for each segmentation mask
    object_pred_kpts = [] # the closest predicted keypoint pixels for each segmentation mask
    object_new_kpts = [] # the closest edited predicted keypoint pixels for each segmentation mask
    object_new_org_box_posit = [] # the original bounding box list position for each segmentation mask

    object_left_distance = [] # the distance between the keypoints and the closest edge pixel for each object in the image
    object_left_edge_position = [] # the closest edge pixel for each keypoint for each object in the image
    object_right_distance = [] # the distance between the keypoints and the closest edge pixel for each object in the image
    object_right_edge_position = [] # the closest edge pixel for each keypoint for each object in the image

    rotation_list = []
    mesial_left = True
    object_count = 0
    # loops the seg_results and finds the edge pixels for each binary mask
    for seg_instances in seg_results:
        object_count += 1
        # gets rotation index
        rotation_index = skew_angle_from_mask(seg_instances, degrees_out=True)
        if show_images:
            print(f"Rotation angle for object {object_count}: {rotation_index:.2f} degrees")
        # rotation_list.append(float(rotation_index))
        if show_images:
            avg_angle = rotation_index - 90
            mask_center = np.array(np.nonzero(seg_instances)).mean(axis=1)
            arrow_length = 100
            arrow_dx = arrow_length * np.cos(np.deg2rad(avg_angle))
            arrow_dy = arrow_length * np.sin(np.deg2rad(avg_angle))
            
            mask_for_arrow = image.copy()
            # overlays seg_instance mask on image
            # Create a blue overlay for the mask
            mask_colour = np.zeros_like(mask_for_arrow)
            mask_colour[..., 0] = (seg_instances * 255).astype(np.uint8)
            mask_for_arrow = cv2.addWeighted(mask_for_arrow, 0.7, mask_colour, 0.3, 0)

            start_point = (int(mask_center[1]), int(mask_center[0]))  # (x, y)
            end_point = (int(mask_center[1] + arrow_dx), int(mask_center[0] + arrow_dy))  # (x, y)
            cv2.arrowedLine(mask_for_arrow, start_point, end_point, (0, 0, 255), 8)
            # Save the image with the arrow instead of showing it
            save_path = os.path.join(arrow_save_loc, f"{os.path.split(image_path)[-1].split('.')[0]}_{str(object_count)}_arrow.png")
            cv2.imwrite(save_path, mask_for_arrow)
        

        # splits seg_instances into two masks from the centre point of mask along the vertical axis with the rotation index
        split_masks_left, split_masks_right = split_masks_deg(seg_instances, rotation_index)        

        # find the edge pixels and converts the edge pixels to binary mask
        edge_instances = cv2.Canny(seg_instances.astype('uint8'), 0, 1)

        # converts 255 vlaue pixels into a list of pixel coordinates
        edge_positions = np.argwhere(edge_instances == 255)
        

        # splits edge_positions into edge_positions_left and edge_positions_right by checking if edge_position is in the left or right mask
        edge_positions_left = edge_positions[np.where(split_masks_left[edge_positions[:, 0], edge_positions[:, 1]] > 0)]
        edge_positions_right = edge_positions[np.where(split_masks_right[edge_positions[:, 0], edge_positions[:, 1]] > 0)]

        edge_positions = np.fliplr(edge_positions)
        # flips the x and y axis of the edge_positions due to formatting issues
        edge_positions_left = np.fliplr(edge_positions_left)
        edge_positions_right = np.fliplr(edge_positions_right)


        closest_edge_position_all = [] # the closest edge pixel for each keypoint for each object in the image
        closest_distance_all = [] # the distance between the keypoints and the closest edge pixel for each object in the image
        closest_distance_avg = [] # the average distance between the keypoints and the closest edge pixel for each object in the image
        closest_pred_pixel_all = [] # the closest predicted keypoint pixel for each object in the image
        closest_original_box_posit = [] # the original bounding box list position for the image
        closest_distance_left_all = [] # the distance between the keypoints and the closest edge pixel for each object in the image
        closest_edge_position_left_all = [] # the closest edge pixel for each keypoint for each object in the image
        closest_edge_position_right_all = [] # the closest predicted keypoint pixel for each object in the image
        closest_distance_right_all = [] # the distance between the keypoints and the closest edge pixel for each object in the image
        # traverses the predicted keypoints and finds the closest edge pixel for each\
        # traverses each object box instance
        for k in range(len(kpt_results)):
            # gets keypoint set and superclass bounding box class
            keypoint = kpt_results[k]
            # gets box class list of keypoint visibility
            box_clss_vis = class_keypoint_visibility_map[int(kpt_box_clss[k].item())]

            # print(keypoint)
            closest_edge_position, closest_edge_position_left, closest_edge_position_right = [], [], [] # the closest edge pixel for each keypoint for the specified object
            closest_distance, closest_distance_left, closest_distance_right = [], [], [] # the distance between the keypoints and the closest edge pixel for the specified object
            closest_pred_pixel = [] # the closest predicted keypoint pixel for the specified object

            # traverses each keypoint in the object
            for p in range(len(keypoint)):
                pixel = keypoint[p]
                # visibility number for current keypoint
                kpt_vis = box_clss_vis[p]
                
                # filters out keypoints with 0 visibility and Furcation/ARR Related Keypoints kpt_clss [7, 8, 9, 10] or keypoint is 0.0 (to handle when keypoints are not detected by the model but should be there)
                if kpt_vis!= 0 and (pixel[0] != 0.0 or pixel[1] != 0.0):
                    if p not in class_ignore:
                        # Calculate the Euclidean distance between the pixel and all edge_positions
                        distances = np.sqrt((edge_positions[:, 0] - pixel[0]) ** 2 + (edge_positions[:, 1] - pixel[1]) ** 2)
                        distances_left = np.sqrt((edge_positions_left[:, 0] - pixel[0]) ** 2 + (edge_positions_left[:, 1] - pixel[1]) ** 2)
                        distances_right = np.sqrt((edge_positions_right[:, 0] - pixel[0]) ** 2 + (edge_positions_right[:, 1] - pixel[1]) ** 2)

                        # splits 
                        # Find the index of the closest edge_position of on the segmentation mask edge for the current detected keypoint
                        closest_index = np.argmin(distances)
                        closest_index_left = np.argmin(distances_left)
                        closest_index_right = np.argmin(distances_right)

                        # saves the closest edge poitions and distances for each keypoint in the current object
                        closest_edge_position.append(edge_positions[closest_index].tolist())
                        closest_distance.append(distances[closest_index])
                        closest_pred_pixel.append(pixel.tolist())
                        closest_edge_position_left.append(edge_positions_left[closest_index_left].tolist())
                        closest_distance_left.append(distances_left[closest_index_left])
                        closest_edge_position_right.append(edge_positions_right[closest_index_right].tolist())
                        closest_distance_right.append(distances_right[closest_index_right])
                    else:
                        # places original kpt classs 7, 8, 9 or 10 keypoints back into the final output
                        closest_edge_position.append([int(x) for x in pixel])
                        closest_distance.append(None)
                        closest_pred_pixel.append(pixel.tolist())
                        closest_edge_position_left.append([int(x) for x in pixel])
                        closest_distance_left.append(None)
                        closest_edge_position_right.append([int(x) for x in pixel])
                        closest_distance_right.append(None)

                else:
                    # adds 0.0 keypoints back into the findal output
                    closest_edge_position.append([0, 0])
                    closest_distance.append(None)
                    closest_pred_pixel.append([0.0, 0.0])
                    closest_edge_position_left.append([0, 0])
                    closest_distance_left.append(None)
                    closest_edge_position_right.append([0, 0])
                    closest_distance_right.append(None)



            # finds the average distance for each each item in closest_distance (each distance between keypoint and closest edge pixel for a given object)
            closest_distance_avg.append(np.mean([x for x in closest_distance if x != None]))

            # saves the closest edge poitions and distances for each keypoint for each object in the image
            closest_distance_all.append(closest_distance)
            closest_edge_position_all.append(closest_edge_position)
            closest_pred_pixel_all.append(closest_pred_pixel)
            # saves the original bounding box list position
            closest_original_box_posit.append(k)

            # left and right closest edge positions and distances
            closest_distance_left_all.append(closest_distance_left)
            closest_edge_position_left_all.append(closest_edge_position_left)
            closest_distance_right_all.append(closest_distance_right)
            closest_edge_position_right_all.append(closest_edge_position_right)
            # closest_class_all.append(closest_class)
        


        # replaces nan values with inf
        closest_distance_avg = np.nan_to_num(closest_distance_avg, nan=np.inf)
        # finds the closest average distance position to match seg mask to keypoint group
        closest_distance_posit = np.argmin(closest_distance_avg)

        # gets distnace and edge pixel all of the closest group of pixels
        closest_distance_all = closest_distance_all[closest_distance_posit]
        closest_edge_position_all = closest_edge_position_all[closest_distance_posit]
        closest_pred_pixel_all = closest_pred_pixel_all[closest_distance_posit]
        closest_distance_avg = closest_distance_avg[closest_distance_posit]
        closest_original_box_posit = closest_original_box_posit[closest_distance_posit]

        # left and right closest
        closest_distance_left_all = closest_distance_left_all[closest_distance_posit]
        closest_edge_position_left_all = closest_edge_position_left_all[closest_distance_posit]
        closest_distance_right_all = closest_distance_right_all[closest_distance_posit]
        closest_edge_position_right_all = closest_edge_position_right_all[closest_distance_posit]
        

        # checks if the current original keypoints are already in the final, if they are, it comapres the average closest distance to choose the most likely seg mask match
        # this scrambles the order of the bounding boxes compared to the model output. This is rectified later.
        if closest_pred_pixel_all in object_pred_kpts:
            # gets the index
            index = object_pred_kpts.index(closest_pred_pixel_all)


            # checks if the new distance is lower than the current distance
            if closest_distance_avg < object_distance[index]:
                object_distance[index] = closest_distance_avg
                object_pred_kpts[index] = closest_pred_pixel_all
                object_new_kpts[index] = closest_edge_position_all
                object_new_org_box_posit[index] = closest_original_box_posit

                object_left_distance[index] = closest_distance_left_all
                object_left_edge_position[index] = closest_edge_position_left_all
                object_right_distance[index] = closest_distance_right_all
                object_right_edge_position[index] = closest_edge_position_right_all
                rotation_list[index] = float(rotation_index)
        else:
            object_distance.append(closest_distance_avg)
            object_pred_kpts.append(closest_pred_pixel_all)
            object_new_kpts.append(closest_edge_position_all)
            object_new_org_box_posit.append(closest_original_box_posit)

            object_left_distance.append(closest_distance_left_all)
            object_left_edge_position.append(closest_edge_position_left_all)
            object_right_distance.append(closest_distance_right_all)
            object_right_edge_position.append(closest_edge_position_right_all)
            rotation_list.append(float(rotation_index))


    if post_process_type == 1:

        # FINDS WHICH SIDE OF THE IMAGE BELONGS TO MESIAL AND DISTAL USING AVERAGE DISTANCES TO ALL POINTS.  REPLACES THE NEW  KEYPOINT WITH THE CLOSEST EDGE PIXEL BASED ON WHICH SIDE THE KEYPOINT BELONGS TO
        # traverses default_left_clss and default_right_clss, compares the closest distance left to closest distance right 
        flip_count = 0 # positive imples tooth order is the correct default side, negative implies the inverted side
        for tooth in range(len(object_left_distance)):
            # checks if the keypoint class is in the left or right side of the mask
            for i in default_left_clss:
                # Compare left and right distances for each keypoint index in default_left_clss
                left_val = object_left_distance[tooth][i]
                right_val = object_right_distance[tooth][i]
                if left_val is not None and right_val is not None:
                    if left_val < right_val:
                        flip_count += 1
                    elif right_val < left_val:
                        flip_count -= 1
            for j in default_right_clss:
                # Compare left and right distances for each keypoint index in default_right_clss
                left_val = object_left_distance[tooth][j] 
                right_val = object_right_distance[tooth][j]
                if left_val is not None and right_val is not None:
                    if left_val < right_val:
                        flip_count -= 1
                    elif right_val < left_val:
                        flip_count += 1

        # get side
        if flip_count >= 0:
            # if the left side is closer, then the left side is the default side
            mesial_left = True
        else:
            # if the right side is closer, then the right side is the default side
            mesial_left = False

        # if medial_left is true, change defaullt_left_clss with object_left_distance and default_right_clss with object_right_distance. invert if False
        for mask in range(len(object_left_distance)):
            for i in default_left_clss:
                if mesial_left:
                    # applies left pixel for default_left_clss to final output
                    object_new_kpts[mask][i] = object_left_edge_position[mask][i]
                else:
                    object_new_kpts[mask][i] = object_right_edge_position[mask][i]
            for i in default_right_clss:
                if mesial_left:
                    # applies right pixel for default_right_clss to final output
                    object_new_kpts[mask][i] = object_right_edge_position[mask][i]
                else:
                    object_new_kpts[mask][i] = object_left_edge_position[mask][i]


    





    # converts object_pred_kpts and object_new_kpts to normalised keypoints
    object_pred_kpts = (np.array(object_pred_kpts) / [width, height]).tolist()
    object_new_kpts = (np.array(object_new_kpts) / [width, height]).tolist()
    original_kpts_old = (np.array(original_kpts_old) / [width, height]).tolist()
    original_kpts_new = (np.array(original_kpts_new) / [width, height]).tolist()

    
    final_rotation_list = [0.0] * len(original_kpts_old)
    # changes the position of object_pred_kpts and object_new_kpts to the original bounding box list positions according to object_new_org_box_posit
    for orig_i in range(len(original_kpts_old)):
        # checks if the original_kpts is in object_new_org_box_posit
        if orig_i in object_new_org_box_posit:
            # change original_kpts_old and original_kpts_new to the new kpt positions
            original_kpts_new[orig_i] = object_new_kpts[object_new_org_box_posit.index(orig_i)]
            original_kpts_old[orig_i] = object_pred_kpts[object_new_org_box_posit.index(orig_i)]
            final_rotation_list[orig_i] = rotation_list[object_new_org_box_posit.index(orig_i)]

    

    object_new_kpts = original_kpts_new
    object_pred_kpts = original_kpts_old
    

    # show results
    if show_images:

        object_pred_kpts_view = object_pred_kpts
        object_new_kpts_view = object_new_kpts

        # converts keypoints to original image size
        object_pred_kpts_view = (np.array(object_pred_kpts_view) * [width, height]).tolist()
        object_new_kpts_view = (np.array(object_new_kpts_view) * [width, height]).tolist()

        result_idx = 0
        # if show_img_verbose:
        for i in range(len(object_pred_kpts_view)):
            result_idx += 1
            fig, ax = plt.subplots()
            ax.imshow(image)
            for seg_instances in seg_results:
                # Normalise overlay if necessary (some libraries load 0-255, others 0-1)
                if seg_instances.max() > 1:
                    seg_instances = seg_instances / 255.0  # Convert to 0-1 range
                # get shape of the image and seg_instances
                seg_instances_rgba = np.zeros((seg_instances.shape[0], seg_instances.shape[1], 4))
                # Assign red color to white pixels
                seg_instances_rgba[..., 0] = 0.0 # Red channel
                seg_instances_rgba[..., 1] = 0.0  # Green channel
                seg_instances_rgba[..., 2] = np.where(seg_instances > 0.9, 1.0, 0.0)  # Blue channel
                # Set alpha (transparency): 0.5 for white pixels, 0 for black pixels
                seg_instances_rgba[..., 3] = np.where(seg_instances > 0.9, 0.5, 0.0)  # 50% opacity for white pixels

                # Overlay the segmentation results with clear background
                ax.imshow(seg_instances_rgba, alpha=0.5, cmap='jet')

            for points in range(len(object_pred_kpts_view[i])):
                # adjusts keypoints back to original image size
                ax.scatter(object_pred_kpts_view[i][points][0], object_pred_kpts_view[i][points][1], c='red')
                # prints the keypoint position number next to the keypoint
                ax.text(object_pred_kpts_view[i][points][0], object_pred_kpts_view[i][points][1], str(points), fontsize=10, color='red')
                    

            # overlay the closest_edge_position
            for points in range(len(object_new_kpts_view[i])):
                ax.scatter(object_new_kpts_view[i][points][0], object_new_kpts_view[i][points][1], c='green')
                # prints the keypoint position number next to the keypoint
                ax.text(object_new_kpts_view[i][points][0], object_new_kpts_view[i][points][1], str(points), fontsize=10, color='green')

            ax.axis('off')
            # plt.show()
            if os.path.exists(os.path.join(args.save_loc, 'visualisations')) == False:
                os.makedirs(os.path.join(args.save_loc, 'visualisations'))
            save_vis = os.path.join(args.save_loc, 'visualisations', img.split('.')[0] + '_' + str(result_idx) + '_matched_mask_kpt.jpg')
            plt.savefig(save_vis)
            plt.close(fig)
        
        fig, ax = plt.subplots()
        ax.imshow(image)
        for seg_instances in seg_results:
            # Normalise overlay if necessary (some libraries load 0-255, others 0-1)
            if seg_instances.max() > 1:
                seg_instances = seg_instances / 255.0  # Convert to 0-1 range
            # get shape of the image and seg_instances
            seg_instances_rgba = np.zeros((seg_instances.shape[0], seg_instances.shape[1], 4))
            # Assign red color to white pixels
            seg_instances_rgba[..., 0] = 0.0 # Red channel
            seg_instances_rgba[..., 1] = 0.0  # Green channel
            seg_instances_rgba[..., 2] = np.where(seg_instances > 0.9, 1.0, 0.0)  # Blue channel
            # Set alpha (transparency): 0.5 for white pixels, 0 for black pixels
            seg_instances_rgba[..., 3] = np.where(seg_instances > 0.9, 0.5, 0.0)  # 50% opacity for white pixels

            # Overlay the segmentation results with clear background
            ax.imshow(seg_instances_rgba, alpha=0.5, cmap='jet')
        for i in range(len(object_pred_kpts_view)):
            for points in range(len(object_pred_kpts_view[i])):
                ax.scatter(object_pred_kpts_view[i][points][0], object_pred_kpts_view[i][points][1], c='red')
            for points in range(len(object_new_kpts_view[i])):
                ax.scatter(object_new_kpts_view[i][points][0], object_new_kpts_view[i][points][1], c='green')
        ax.axis('off')

        # save image
        if os.path.exists(os.path.join(args.save_loc, 'visualisations')) == False:
            os.makedirs(os.path.join(args.save_loc, 'visualisations'))
        save_vis = os.path.join(args.save_loc, 'visualisations', img.split('.')[0] + '_final_matches_all.jpg')
        plt.savefig(save_vis)
        plt.close(fig)

    

    return(object_new_kpts, object_pred_kpts, object_distance, width, height, final_rotation_list)