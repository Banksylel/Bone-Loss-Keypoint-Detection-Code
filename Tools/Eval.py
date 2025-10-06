from collections import defaultdict
import cv2
import copy
import torch
import math
import os
from ultralytics import YOLO
import numpy as np
from Tools.Post_Processing import post_process
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from All_Metrics import calculate_metrics, calculate_furcation_metrics, add_to_metrics, find_intersection, calculate_pbl, calculate_bone_loss, calculate_iou, prck, nme, get_nmse


# removes none and nan values from results list
def filter_none(lst, replace_val = 0.0):
    """
    Filters out None and NaN values from a list or 2D list and replaces them with a specified value.

    Returns the filtered list.
    """
    if isinstance(lst[0], list):
        return [[replace_val if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in row] for row in lst]
    else:
        return [replace_val if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in lst]



# overleaf 
def Evaluate_Model(weight_path_orig, weights_orig, rotate_loc_orig, pred_imgs_orig, pred_lbls_orig, best_seg_weights, kpt_classes, args):
    """
    Traverses all folds and validation data to evaluate the model using various metrics.

    Returns a dictionary containing all calculated metrics.
    """
    # metrics 
    prck_thresh_list = {}
    cur_thresh = 0.5
    while cur_thresh > 0.04:
        prck_thresh_list[cur_thresh] = []
        cur_thresh -= 0.05
        cur_thresh = round(cur_thresh, 2) 
    nme_list, prck_005_list, prck_025_list, prck_05_list, f1_pbl_m_list, f1_pbl_d_list, prec_pbl_m_list, prec_pbl_d_list, spec_pbl_m_list, spec_pbl_d_list, sens_pbl_m_list, sens_pbl_d_list =  [], [], [], [], [], [], [], [], [], [], [], []
    spec_f_list, sens_f_list, f1_f_list, prec_f_list = [[],[]], [[],[]], [[],[]], [[],[]]
    targ_rotate_index_list_fold, pred_rotate_index_list_fold, target_class_index_list_fold = [], [], []
    
    # loops though all folds in the fold folder
    for fold in range(args.folds):
        weight_path = os.path.join(weight_path_orig, weights_orig[fold])
        if args.test_set == False:
            rotate_loc = os.path.join(rotate_loc_orig, 'f'+str(fold), 'val', 'labels')
            pred_imgs = os.path.join(pred_imgs_orig, 'f'+str(fold), 'val', 'images')
            pred_lbls = os.path.join(pred_lbls_orig, 'f'+str(fold), 'val', 'labels')
        else:
            rotate_loc = rotate_loc_orig
            pred_imgs = pred_imgs_orig
            pred_lbls = pred_lbls_orig

        # initialise kpt model
        model_kpts = YOLO(weight_path)
        # initialise seg model
        model_seg = YOLO(best_seg_weights)

        prck_thresh_num = {}
        prck_thresh_denom = {}
        cur_thresh = 0.5
        while cur_thresh > 0.04:
            prck_thresh_num[cur_thresh] = []
            prck_thresh_denom[cur_thresh] = []
            for met in range(len(kpt_classes)):
                prck_thresh_num[cur_thresh].append(0)
                prck_thresh_denom[cur_thresh].append(0)
            cur_thresh -= 0.05
            cur_thresh = round(cur_thresh, 2) 


        norm_errors = []
        # populates pdj metrics
        for met in range(len(kpt_classes)):
            norm_errors.append([])
            
        metrics_oks = []
        pbl_tp_m, pbl_fp_m, pbl_tn_m, pbl_fn_m = [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]
        pbl_tp_d, pbl_fp_d, pbl_tn_d, pbl_fn_d = [0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]
        fur_tp, fur_fp, fur_tn, fur_fn = [0,0], [0,0], [0,0], [0,0]
        targ_rotate_index_list, pred_rotate_index_list, target_class_index_list = [], [], []
        for img in os.listdir(pred_imgs):
            
            image_path = os.path.join(pred_imgs, img)

            # loads image
            img_object = cv2.imread(image_path)

            pred = model_kpts.predict(image_path, save=False, imgsz=args.image_size, iou=args.pred_kpts_iou, conf=args.pred_kpts_conf)

            # applies boxes and keypoints to the image
            if args.view_images:
                img_object_vis = copy.deepcopy(img_object)

                img_w_new = img_object.shape[1]
                img_h_new = img_object.shape[0]

                box_pred_img = pred[0].boxes.xywhn
                kpt_pred_img = pred[0].keypoints.xyn
                pred_box_conf = pred[0].boxes.conf
                pred_box_clss = pred[0].boxes.cls

                # Convert YOLO format (normalized center x, y, width, height) to pixel format (top-left x, y, width, height)
                # Converts from tensor to numpy array
                box_pred_img = box_pred_img.cpu().numpy() if hasattr(box_pred_img, "cpu") else np.array(box_pred_img)
                # Denormalize using new image size (already normalized, just scale)
                box_pred_img[:, 0] = box_pred_img[:, 0] * img_w_new  # center x
                box_pred_img[:, 1] = box_pred_img[:, 1] * img_h_new  # center y
                box_pred_img[:, 2] = box_pred_img[:, 2] * img_w_new  # width
                box_pred_img[:, 3] = box_pred_img[:, 3] * img_h_new  # height
                # Convert center x,y to top-left x,y
                box_pred_img[:, 0] = box_pred_img[:, 0] - box_pred_img[:, 2] / 2
                box_pred_img[:, 1] = box_pred_img[:, 1] - box_pred_img[:, 3] / 2
                box_pred_img = box_pred_img.tolist()

                # Convert keypoints from normalized to pixel coordinates and scale to new image size
                kpt_pred_img = kpt_pred_img.cpu().numpy() if hasattr(kpt_pred_img, "cpu") else np.array(kpt_pred_img)
                kpt_pred_img = (np.array(kpt_pred_img) * [img_w_new, img_h_new]).tolist()


                for box_itr in range(len(box_pred_img)):
                    box_pred = box_pred_img[box_itr]
                    img_object_vis = cv2.rectangle(img_object_vis, (int(box_pred[0]), int(box_pred[1])), (int(box_pred[0]+box_pred[2]), int(box_pred[1]+box_pred[3])), (255,0,0), 2)
                    # Display confidence value at the top-left of the predicted box
                    conf_text = f"{pred_box_conf[box_itr]:.2f}"
                    conf_text = f"{int(pred_box_clss[box_itr])}:{pred_box_conf[box_itr]:.2f}"
                    img_object_vis = cv2.putText(img_object_vis,conf_text,(int(box_pred[0]), int(box_pred[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)

                for kpt_pred in kpt_pred_img:
                    for kpt in kpt_pred:
                        img_object_vis = cv2.circle(img_object_vis, (int(kpt[0]), int(kpt[1])), 5, (255,0,0), -1)

                # save image
                if os.path.exists(os.path.join(args.save_loc, 'visualisations')) == False:
                    os.makedirs(os.path.join(args.save_loc, 'visualisations'))
                save_vis = os.path.join(args.save_loc, 'visualisations', img.split('.')[0] + '_direct_out.jpg')
                cv2.imwrite(save_vis, img_object_vis)
            
            # gets rotation box data
            rotate_file = os.path.join(rotate_loc, img.split('.')[0] + '.txt')
            with open(rotate_file, 'r') as f:
                rotate_target = f.read().splitlines()
                rotate_target = [i.split(' ') for i in rotate_target]
            # gets degrees of rotation from rotate_target
            rotate_deg = [i[-1] for i in rotate_target]
            # converts None to 0.0
            for i in range(len(rotate_deg)):
                if rotate_deg[i] == 'None':
                    rotate_deg[i] = 0.0
                else:
                    rotate_deg[i] = float(rotate_deg[i])

            # post processes the predicted keypoints
            if args.post_process_kpts:
                new_pred, old_kpts, _ ,img_width, img_height, rotate_pred = post_process(model_seg, pred, image_path, args)
                # converts new_pred into a tensor on the same decvice as pred[0].keypoints.xyn
                pred_kpt_coords = torch.tensor(new_pred, device=pred[0].keypoints.xyn.device)
            else:
                _, _, _, img_width, img_height, rotate_pred = post_process(model_seg, pred, image_path, args, img_dims_only=True)
                

                # converts new_pred into a tensor on the same decvice as pred[0].keypoints.xyn
                pred_kpt_coords = pred[0].keypoints.xyn

            # finds and opens the corresponding label file
            lbl_file = img.split('.')[0] + '.txt'
            lbl_path = os.path.join(pred_lbls, lbl_file)
            with open(lbl_path, 'r') as f:
                lbl = f.read().splitlines()
            # extracts the box and keypoint information from the label file and places them in the same format as pred box_coords and kpt_coords
            lbl_box_clss = []
            lbl_box_coords = []
            lbl_kpt_coords = []
            lbl_kpt_visibility = []
            for l in lbl:
                l = l.split(' ')

                lbl_box_clss.append(int(l[0]))
                lbl_box_coords.append([float(l[1]), float(l[2]), float(l[3]), float(l[4])])
                lbl_kpt_coords.append([[float(l[5]), float(l[6])], [float(l[8]), float(l[9])], [float(l[11]), float(l[12])], [float(l[14]), float(l[15])], [float(l[17]), float(l[18])], [float(l[20]), float(l[21])], [float(l[23]), float(l[24])], [float(l[26]), float(l[27])], [float(l[29]), float(l[30])], [float(l[32]), float(l[33])], [float(l[35]), float(l[36])]])
                lbl_kpt_visibility.append([int(float(l[7])), int(float(l[10])), int(float(l[13])), int(float(l[16])), int(float(l[19])), int(float(l[22])), int(float(l[25])), int(float(l[28])), int(float(l[31])), int(float(l[34])), int(float(l[37]))])

            

            pred_box_clss = pred[0].boxes.cls
            pred_box_conf = pred[0].boxes.conf
            pred_box_coords = pred[0].boxes.xywhn
            pred_kpt_conf = pred[0].keypoints.conf

            # converts pred box coords center point x,y to top left corner x,y
            for i in range(len(pred_box_coords)):
                pred_box_coords[i][0] = pred_box_coords[i][0] - pred_box_coords[i][2]/2
                pred_box_coords[i][1] = pred_box_coords[i][1] - pred_box_coords[i][3]/2
                # de-nomalises the box coords
                pred_box_coords[i][0] = pred_box_coords[i][0] * img_width
                pred_box_coords[i][1] = pred_box_coords[i][1] * img_height
                pred_box_coords[i][2] = pred_box_coords[i][2] * img_width
                pred_box_coords[i][3] = pred_box_coords[i][3] * img_height
                for j in range(len(pred_kpt_coords[i])):
                    pred_kpt_coords[i][j][0] = pred_kpt_coords[i][j][0] * img_width
                    pred_kpt_coords[i][j][1] = pred_kpt_coords[i][j][1] * img_height

            # converts target box coords and kpt coords into coco format (non normaised, xy(top left corner) wh and xy)
            for i in range(len(lbl_box_coords)):
                # moves x, y from center to top left corner
                lbl_box_coords[i][0] = lbl_box_coords[i][0] - lbl_box_coords[i][2]/2
                lbl_box_coords[i][1] = lbl_box_coords[i][1] - lbl_box_coords[i][3]/2
                # de-nomalises the box coords and kpt coords
                lbl_box_coords[i][0] = lbl_box_coords[i][0] * img_width
                lbl_box_coords[i][1] = lbl_box_coords[i][1] * img_height
                lbl_box_coords[i][2] = lbl_box_coords[i][2] * img_width
                lbl_box_coords[i][3] = lbl_box_coords[i][3] * img_height
                for j in range(len(lbl_kpt_coords[i])):
                    lbl_kpt_coords[i][j][0] = lbl_kpt_coords[i][j][0] * img_width
                    lbl_kpt_coords[i][j][1] = lbl_kpt_coords[i][j][1] * img_height

            # convert tensor array to lists
            pred_box_clss = pred_box_clss.tolist()
            pred_box_conf = pred_box_conf.tolist()
            pred_box_coords = pred_box_coords.tolist()
            pred_kpt_conf = pred_kpt_conf.tolist()
            pred_kpt_coords = pred_kpt_coords.tolist()


            # applies boxes and keypoints to the image
            if args.view_images:
                img_object_vis = copy.deepcopy(img_object)
                img_object_vis_boxes = copy.deepcopy(img_object)
                # converts the coordinates back into the image object size
                img_w_new = img_object.shape[1]
                img_h_new = img_object.shape[0]
                # box_pred_img = pred_box_coords
                box_pred_img = (np.array(pred_box_coords) / [img_width, img_height, img_width, img_height]).tolist()
                box_pred_img = (np.array(box_pred_img) * [img_w_new, img_h_new, img_w_new, img_h_new]).tolist()
                box_lbl_img = (np.array(lbl_box_coords) / [img_width, img_height, img_width, img_height]).tolist()
                box_lbl_img = (np.array(box_lbl_img) * [img_w_new, img_h_new, img_w_new, img_h_new]).tolist()
                kpt_pred_img = (np.array(pred_kpt_coords) / [img_width, img_height]).tolist()
                kpt_pred_img = (np.array(kpt_pred_img) * [img_w_new, img_h_new]).tolist()
                kpt_lbl_img = (np.array(lbl_kpt_coords) / [img_width, img_height]).tolist()
                kpt_lbl_img = (np.array(kpt_lbl_img) * [img_w_new, img_h_new]).tolist()

                if not len(box_pred_img) == len(kpt_pred_img):
                    print("Mismatch between box and keypoint predictions")

                for box_itr in range(len(box_pred_img)):
                    box_pred = box_pred_img[box_itr]
                    img_object_vis = cv2.rectangle(img_object_vis, (int(box_pred[0]), int(box_pred[1])), (int(box_pred[0]+box_pred[2]), int(box_pred[1]+box_pred[3])), (255,0,0), 2)
                    # Display confidence value at the top-left of the predicted box
                    conf_text = f"{int(pred_box_clss[box_itr])}:{pred_box_conf[box_itr]:.2f}"
                    img_object_vis = cv2.putText(img_object_vis,conf_text,(int(box_pred[0]), int(box_pred[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)
                    img_object_vis_boxes = cv2.rectangle(img_object_vis_boxes, (int(box_pred[0]), int(box_pred[1])), (int(box_pred[0]+box_pred[2]), int(box_pred[1]+box_pred[3])), (255,0,0), 2)
                    img_object_vis_boxes = cv2.putText(img_object_vis_boxes,conf_text,(int(box_pred[0]), int(box_pred[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 0, 0), 2, cv2.LINE_AA)
                # for kpt_pred in kpt_pred_img:
                    for kpt in kpt_pred_img[box_itr]:
                        img_object_vis = cv2.circle(img_object_vis, (int(kpt[0]), int(kpt[1])), 5, (255,0,0), -1)
                    if os.path.exists(os.path.join(args.save_loc, 'visualisations')) == False:
                        os.makedirs(os.path.join(args.save_loc, 'visualisations'))
                    save_vis = os.path.join(args.save_loc, 'visualisations', img.split('.')[0] + "_object_" + str(box_itr) + '_processed_out.jpg')
                    cv2.imwrite(save_vis, img_object_vis)

                if not len(box_lbl_img) == len(kpt_lbl_img):
                    print("Mismatch between box and keypoint labels")

                for box_lbl in box_lbl_img:
                    img_object_vis = cv2.rectangle(img_object_vis, (int(box_lbl[0]), int(box_lbl[1])), (int(box_lbl[0]+box_lbl[2]), int(box_lbl[1]+box_lbl[3])), (0,255,0), 2)
                    img_object_vis_boxes = cv2.rectangle(img_object_vis_boxes, (int(box_lbl[0]), int(box_lbl[1])), (int(box_lbl[0]+box_lbl[2]), int(box_lbl[1]+box_lbl[3])), (0,255,0), 2)
                for kpt_lbl in kpt_lbl_img:
                    for kpt in kpt_lbl:
                        img_object_vis = cv2.circle(img_object_vis, (int(kpt[0]), int(kpt[1])), 5, (0,255,0), -1)
                # save image
                if os.path.exists(os.path.join(args.save_loc, 'visualisations')) == False:
                    os.makedirs(os.path.join(args.save_loc, 'visualisations'))
                save_vis = os.path.join(args.save_loc, 'visualisations', img.split('.')[0] + '_processed_out.jpg')
                cv2.imwrite(save_vis, img_object_vis)

                if os.path.exists(os.path.join(args.save_loc, 'visualisations/boxes')) == False:
                    os.makedirs(os.path.join(args.save_loc, 'visualisations/boxes'))
                save_vis2 = os.path.join(args.save_loc, 'visualisations/boxes', img.split('.')[0] + '_boxes_processed_out.jpg')
                cv2.imwrite(save_vis2, img_object_vis_boxes)


            # match pred and lbl boxes regardless of class
            matched_indexes = []

            # matched_indexes shows the pred_box_coords index number (value) at the position of the lbl_box_coords index number (index of matched_indexes) and adds rectangle to images with text linked to the box class
            for i, lbl_box in enumerate(lbl_box_coords):

                best_iou = 0
                best_j = -1
                for j, pred_box in enumerate(pred_box_coords):
                    iou = calculate_iou(lbl_box, pred_box)
                    if iou >= args.box_iou_thresh and iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j != -1:
                    matched_indexes.append(best_j)
                else:
                    matched_indexes.append(None)


            
            # uses matched_indexes to create two lists for pred and lbl keypoints that are matched as a 2d list, does not include None or non matched keypoints
            matched_pred_kpts = []
            matched_lbl_kpts = []
            matched_lbl_kpt_visibility = []
            matched_rotate_target = []
            matched_rotate_pred = []
            matched_target_class = []

            # matched_kpt_vis = []
            for i, match in enumerate(matched_indexes):
                if match != None:
                    # target lbl items remain the same
                    matched_lbl_kpts.append(lbl_kpt_coords[i])
                    # pred items are sorted to match the target lbl items
                    matched_pred_kpts.append(pred_kpt_coords[match])
                    matched_lbl_kpt_visibility.append(lbl_kpt_visibility[i])
                    # box rotation data
                    matched_rotate_target.append(rotate_deg[i])
                    matched_rotate_pred.append(rotate_pred[match])
                    matched_target_class.append(lbl_box_clss[i])


            remaining_lbl_kpts = []
            remaining_pred_kpts = []
            remaining_lbl_vis = []
            remaining_pred_vis = []

            # removes items in pred_kpt_coords if they are in matched_pred_kpts
            for i in range(len(pred_kpt_coords)):
                if pred_kpt_coords[i] not in matched_pred_kpts:
                    remaining_pred_kpts.append(pred_kpt_coords[i])
                    temp = []
                    for j in range(len(pred_kpt_coords[i])):
                        if float(pred_kpt_coords[i][j][0]) == 0.0:
                            temp.append(0)
                        else:
                            temp.append(1)
                    remaining_pred_vis.append(temp)

            for i in range(len(lbl_kpt_coords)):
                if lbl_kpt_coords[i] not in matched_lbl_kpts:
                    remaining_lbl_kpts.append(lbl_kpt_coords[i])
                    remaining_lbl_vis.append(lbl_kpt_visibility[i])


            # finds the average box width and height from lbl_box_coords
            total_width = 0
            total_height = 0
            for box in lbl_box_coords:
                total_width += box[2]
                total_height += box[3]
            average_box_width = total_width / len(lbl_box_coords)
            average_box_height = total_height / len(lbl_box_coords)

            
            for thresh in prck_thresh_num.keys():
                prck_thresh_num[thresh], prck_thresh_denom[thresh] = prck([matched_lbl_kpts, matched_pred_kpts, matched_lbl_kpt_visibility], [remaining_lbl_kpts, remaining_pred_kpts, remaining_lbl_vis, remaining_pred_vis], average_box_width, average_box_height, prck_thresh_num[thresh], prck_thresh_denom[thresh], thresh, args.include_fp_fn_prck)
            
            # primary, secondary, avg_target_box_height, avg_target_box_width, norm_error, include_non_matched=False
            norm_errors = nme([matched_lbl_kpts, matched_pred_kpts, matched_lbl_kpt_visibility], [remaining_lbl_kpts, remaining_pred_kpts, remaining_lbl_vis, remaining_pred_vis], average_box_width, average_box_height, norm_errors, args.include_fp_fn_nme)

            
            # adds rotation indexes to final lists
            targ_rotate_index_list.append(matched_rotate_target)
            pred_rotate_index_list.append(matched_rotate_pred)
            target_class_index_list.append(matched_target_class)
            # calculates the bone loss and furcation involvement for each tooth (matched_rotate_target, matched_lbl_kpts, matched_pred_kpts, matched_lbl_kpt_visibility)
            # traverses matched_lbl_kpts and calculates the bone loss for each tooth
            for posit in range(len(matched_lbl_kpts)):
                # extracts all keypoints and relevant information for the current tooth
                lbl_kpts = matched_lbl_kpts[posit]
                # print(lbl_kpts)
                pred_kpts = matched_pred_kpts[posit]
                lbl_vis = matched_lbl_kpt_visibility[posit]
                # if args.test_set == False:
                rotate_index_target = matched_rotate_target[posit]
                rotate_index_pred = matched_rotate_pred[posit]
                # converts lists to an np array of floats
                lbl_kpts = np.array(lbl_kpts).astype(float)
                pred_kpts = np.array(pred_kpts).astype(float)
                
                # changes non visible keypoint to none for calculating bone loss
                if lbl_vis[2] == 0:
                    lbl_kpts[2][0], lbl_kpts[2][1] = None, None
                    pred_kpts[2][0], pred_kpts[2][1] = None, None
                if lbl_vis[5] == 0:
                    lbl_kpts[5][0], lbl_kpts[5][1] = None, None
                    pred_kpts[5][0], pred_kpts[5][1] = None, None
                if lbl_vis[6] == 0:
                    lbl_kpts[6][0], lbl_kpts[6][1] = None, None
                    pred_kpts[6][0], pred_kpts[6][1] = None, None
                img_object = cv2.imread(os.path.join(pred_imgs, img))
                # calculates percentage of bone loss using cej, bone level and root level keypoint distances along the angel of rotation
                pbl_lbl_m, img_object = calculate_bone_loss(lbl_kpts[0], lbl_kpts[1], lbl_kpts[2], lbl_kpts[5], lbl_kpts[6], rotate_index_target, img_object) # pbl_m target
                pbl_lbl_d, img_object = calculate_bone_loss(lbl_kpts[3], lbl_kpts[4], lbl_kpts[2], lbl_kpts[5], lbl_kpts[6], rotate_index_target, img_object) # pbl_d target
                pbl_pred_m, img_object = calculate_bone_loss(pred_kpts[0], pred_kpts[1], pred_kpts[2], pred_kpts[5], pred_kpts[6], rotate_index_pred, img_object) # pbl_m pred 
                pbl_pred_d, img_object = calculate_bone_loss(pred_kpts[3], pred_kpts[4], pred_kpts[2], pred_kpts[5], pred_kpts[6], rotate_index_pred, img_object) # pbl_d pred

                # chooses the highest bone loss value for each tooth
                pbl_m = max(pbl_lbl_m)
                pbl_d = max(pbl_lbl_d)
                pbl_m_pred = max(pbl_pred_m)
                pbl_d_pred = max(pbl_pred_d)

                # applies pbl tp, fp, tn, fn to the metrics
                pbl_tp_m, pbl_fp_m, pbl_tn_m, pbl_fn_m = add_to_metrics(pbl_m, pbl_m_pred, pbl_tp_m, pbl_fp_m, pbl_tn_m, pbl_fn_m)
                pbl_tp_d, pbl_fp_d, pbl_tn_d, pbl_fn_d = add_to_metrics(pbl_d, pbl_d_pred, pbl_tp_d, pbl_fp_d, pbl_tn_d, pbl_fn_d)

                # checks if the current tooth has furcation area in target visibility
                if lbl_vis[7] != 0 or lbl_vis[8] != 0 or lbl_vis[9] != 0:
                    # calculates the furcation involvement for each tooth if present
                    fur_tp, fur_fp, fur_tn, fur_fn = calculate_furcation_metrics(lbl_kpts[7], lbl_kpts[8], lbl_kpts[9], pred_kpts[7], pred_kpts[8], pred_kpts[9], average_box_width, average_box_height, fur_tp, fur_fp, fur_tn, fur_fn, args.furcation_dist_thresh)

                
        # calulate all metrics. PPV, sensitivity, specificity, precision, F1 score. average over all folds and standard deviation
        # furcation
        for clss in range(2):
            sens_f, spec_f, prec_f, f1_f = calculate_metrics(fur_tp[clss], fur_fp[clss], fur_tn[clss], fur_fn[clss])
            # adds to fold list
            sens_f_list[clss].append(sens_f)
            spec_f_list[clss].append(spec_f)
            prec_f_list[clss].append(prec_f)
            f1_f_list[clss].append(f1_f)

        

        sens_pbl_m, spec_pbl_m, prec_pbl_m, f1_pbl_m = [],[],[],[]
        sens_pbl_d, spec_pbl_d, prec_pbl_d, f1_pbl_d = [],[],[],[]
        # precentage bone loss. stores as [healty, mild, moderate, severe], ..
        for cls in range(4):
            sens, spec, prec, f1 = calculate_metrics(pbl_tp_m[cls], pbl_fp_m[cls], pbl_tn_m[cls], pbl_fn_m[cls])
            sens_pbl_m.append(sens)
            spec_pbl_m.append(spec)
            prec_pbl_m.append(prec)
            f1_pbl_m.append(f1)
            sens, spec, prec, f1  = calculate_metrics(pbl_tp_d[cls], pbl_fp_d[cls], pbl_tn_d[cls], pbl_fn_d[cls])
            sens_pbl_d.append(sens)
            spec_pbl_d.append(spec)
            prec_pbl_d.append(prec)
            f1_pbl_d.append(f1)

        # adds average metrics to results
        sens_pbl_d_list.append([np.nanmean(np.array(sens_pbl_d, dtype=float))] + sens_pbl_d)
        spec_pbl_d_list.append([np.nanmean(np.array(spec_pbl_d, dtype=float))] + spec_pbl_d)
        prec_pbl_d_list.append([np.nanmean(np.array(prec_pbl_d, dtype=float))] + prec_pbl_d)
        f1_pbl_d_list.append([np.nanmean(np.array(f1_pbl_d, dtype=float))] + f1_pbl_d)
        sens_pbl_m_list.append([np.nanmean(np.array(sens_pbl_m, dtype=float))] + sens_pbl_m)
        spec_pbl_m_list.append([np.nanmean(np.array(spec_pbl_m, dtype=float))] + spec_pbl_m)
        prec_pbl_m_list.append([np.nanmean(np.array(prec_pbl_m, dtype=float))] + prec_pbl_m)
        f1_pbl_m_list.append([np.nanmean(np.array(f1_pbl_m, dtype=float))] + f1_pbl_m)

        
        for thresh in prck_thresh_num.keys():
            prck_thresh = []
            prck_thresh.append(sum(prck_thresh_num[thresh])/sum(prck_thresh_denom[thresh]))


            for clss in range(len(prck_thresh_num[thresh])):
                if prck_thresh_denom[thresh][clss] == 0:
                    prck_thresh.append(0.0)
                else:
                    prck_thresh.append(prck_thresh_num[thresh][clss]/prck_thresh_denom[thresh][clss])

            prck_thresh_list[thresh].append(prck_thresh)

        # adds rotation indexes to final list for each fold
        targ_rotate_index_list_fold.append(targ_rotate_index_list)
        pred_rotate_index_list_fold.append(pred_rotate_index_list)
        target_class_index_list_fold.append(target_class_index_list)

        
        # finds the average normalised mean error for each keypoint
        class_nme = []
        # fonds nme error for each class
        for i in range(len(norm_errors)):
            class_nme.append(np.nanmean(np.array(norm_errors[i], dtype=float)))

        # sores as f0[average, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10], f1 single values
        nme_list.append([np.nanmean(np.array(class_nme, dtype=float))] + class_nme)
        

    sens_f_list = filter_none(sens_f_list)
    spec_f_list = filter_none(spec_f_list)
    prec_f_list = filter_none(prec_f_list)
    f1_f_list = filter_none(f1_f_list)

    targ_rotate_index_list_fold = filter_none(targ_rotate_index_list_fold)
    pred_rotate_index_list_fold = filter_none(pred_rotate_index_list_fold)
    target_class_index_list_fold = filter_none(target_class_index_list_fold)

    print('########## Rotation Index Normalised Mean Squared Error ##########')
    # normalised mean squared error for rotate_index_target and rotate_index_pred (nmse_list_class)
    nmse_list_class = get_nmse(targ_rotate_index_list_fold, pred_rotate_index_list_fold, target_class_index_list_fold)
    print('########Average Normalised Mean Squared Error ########')
    print(np.nanmean(nmse_list_class), " +/- ", np.std(nmse_list_class))
    rotation_metrics = pd.DataFrame({'Class':['All'], 'NMSE_Mean': [np.nanmean(np.array(nmse_list_class, dtype=float)).item()], 'NMSE_Std': [np.std(np.array(nmse_list_class, dtype=float)).item()]})

    print('######## Class-wise Normalised Mean Squared Error ########')
    nmse_array = np.array(nmse_list_class, dtype=float)
    num_classes = nmse_array.shape[1] if nmse_array.ndim > 1 else 0
    for class_idx in range(num_classes):
        class_nmse = nmse_array[:, class_idx]
        print(f'Class {class_idx}: {np.nanmean(class_nmse)} +/- {np.std(class_nmse)}')

        new_row = pd.DataFrame({
            'Class': [class_idx],
            'NMSE_Mean': [np.nanmean(np.array(class_nmse, dtype=float)).item()],
            'NMSE_Std': [np.std(np.array(class_nmse, dtype=float)).item()]
        })
        rotation_metrics = pd.concat([rotation_metrics, new_row], ignore_index=True)

    rotation_metrics.to_csv(os.path.join(args.save_loc, 'rotation_metrics.csv'), index=False)

    print('\n')
    print('\n')


    print('########## Furcation Involvement ##########')
    print('##### AVERAGE #####')
    print('Precision: ', np.nanmean(np.array(prec_f_list, dtype=float)), ' +/- ', np.std(np.array(prec_f_list, dtype=float)))
    print('Recall: ', np.nanmean(np.array(sens_f_list, dtype=float)), ' +/- ', np.std(np.array(sens_f_list, dtype=float)))
    print('F1 Score: ', np.nanmean(np.array(f1_f_list, dtype=float)), ' +/- ', np.std(np.array(f1_f_list, dtype=float)))
    print('Specificity: ', np.nanmean(np.array(spec_f_list, dtype=float)), ' +/- ', np.std(np.array(spec_f_list, dtype=float)))

    print('\n')
    print('##### CLASS-WISE #####')
    for clss in range(len(sens_f_list)):
        print('Class ', clss)
        print('Precision: ', np.nanmean(np.array(prec_f_list[clss], dtype=float)), ' +/- ', np.std(np.array(prec_f_list[clss], dtype=float)))
        print('Recall: ', np.nanmean(np.array(sens_f_list[clss], dtype=float)), ' +/- ', np.std(np.array(sens_f_list[clss], dtype=float)))
        print('F1 Score: ', np.nanmean(np.array(f1_f_list[clss], dtype=float)), ' +/- ', np.std(np.array(f1_f_list[clss], dtype=float)))
        print('Specificity: ', np.nanmean(np.array(spec_f_list[clss], dtype=float)), ' +/- ', np.std(np.array(spec_f_list[clss], dtype=float)))
        print('\n')

    print('\n')
    print('\n')

    # saves furcation metrics for each class as csv in args.save_loc
    furcation_metrics = pd.DataFrame({'Class':['All'], 'Precision_Mean': [np.nanmean(np.array(prec_f_list, dtype=float)).item()], 'Precision_Std': [np.std(np.array(prec_f_list, dtype=float)).item()], 'Recall_Mean': [np.nanmean(np.array(sens_f_list, dtype=float)).item()], 'Recall_Std': [np.std(np.array(sens_f_list, dtype=float)).item()], 'F1_Score_Mean': [np.nanmean(np.array(f1_f_list, dtype=float)).item()], 'F1_Score_Std': [np.std(np.array(f1_f_list, dtype=float)).item()], 'Specificity_Mean': [np.nanmean(np.array(spec_f_list, dtype=float)).item()], 'Specificity_Std': [np.std(np.array(spec_f_list, dtype=float)).item()]})
    for clss in range(len(sens_f_list)):
        new_row = pd.DataFrame({
            'Class': [clss],
            'Precision_Mean': [np.nanmean(np.array(prec_f_list[clss], dtype=float)).item()],
            'Precision_Std': [np.std(np.array(prec_f_list[clss], dtype=float)).item()],
            'Recall_Mean': [np.nanmean(np.array(sens_f_list[clss], dtype=float)).item()],
            'Recall_Std': [np.std(np.array(sens_f_list[clss], dtype=float)).item()],
            'F1_Score_Mean': [np.nanmean(np.array(f1_f_list[clss], dtype=float)).item()],
            'F1_Score_Std': [np.std(np.array(f1_f_list[clss], dtype=float)).item()],
            'Specificity_Mean': [np.nanmean(np.array(spec_f_list[clss], dtype=float)).item()],
            'Specificity_Std': [np.std(np.array(spec_f_list[clss], dtype=float)).item()]
        })
        furcation_metrics = pd.concat([furcation_metrics, new_row], ignore_index=True)
    furcation_metrics.to_csv(os.path.join(args.save_loc, 'furcation_metrics.csv'), index=False)


    
    # reformats the prck, nme and pbl metrics from [f0[average, c0, c1, ...] , f1[average,c2,c3, ...], ...] to [average[f0, f1, ...], c0[f0, f1, ...], c1[f0, f1, ...], ...]
    prck_thresh_list = {k: list(map(list, zip(*v))) for k, v in prck_thresh_list.items()}
    nme_list = list(map(list, zip(*nme_list)))
    # if args.test_set == False:
    sens_pbl_m_list = list(map(list, zip(*sens_pbl_m_list)))
    spec_pbl_m_list = list(map(list, zip(*spec_pbl_m_list)))
    prec_pbl_m_list = list(map(list, zip(*prec_pbl_m_list)))    
    f1_pbl_m_list = list(map(list, zip(*f1_pbl_m_list)))
    sens_pbl_d_list = list(map(list, zip(*sens_pbl_d_list)))
    spec_pbl_d_list = list(map(list, zip(*spec_pbl_d_list)))
    prec_pbl_d_list = list(map(list, zip(*prec_pbl_d_list)))
    f1_pbl_d_list = list(map(list, zip(*f1_pbl_d_list)))


    prck_thresh_list = {k: filter_none(v) for k, v in prck_thresh_list.items()}

    nme_list = filter_none(nme_list, replace_val=1.0)
    sens_pbl_m_list = filter_none(sens_pbl_m_list)
    spec_pbl_m_list = filter_none(spec_pbl_m_list)
    prec_pbl_m_list = filter_none(prec_pbl_m_list)
    f1_pbl_m_list = filter_none(f1_pbl_m_list)
    sens_pbl_d_list = filter_none(sens_pbl_d_list)
    spec_pbl_d_list = filter_none(spec_pbl_d_list)
    prec_pbl_d_list = filter_none(prec_pbl_d_list)
    f1_pbl_d_list = filter_none(f1_pbl_d_list)



    clss_list = ['Healthy(<0.15)', 'Mild(0.15-0.33)', 'Moderate(0.33-0.66)', 'Severe(>0.66)']
    print('########## precentage bone loss ##########')
    print('###### Mesial Bone Loss ######')
    print('## Average PBL ##')
    print('Precision: ', np.nanmean(np.array(prec_pbl_m_list[0], dtype=float)), ' +/- ', np.std(np.array(prec_pbl_m_list[0], dtype=float)))
    print('Recall: ', np.nanmean(np.array(sens_pbl_m_list[0], dtype=float)), ' +/- ', np.std(np.array(sens_pbl_m_list[0], dtype=float)))
    print('F1 Score: ', np.nanmean(np.array(f1_pbl_m_list[0], dtype=float)), ' +/- ', np.std(np.array(f1_pbl_m_list[0], dtype=float)))
    print('Specificity: ', np.nanmean(np.array(spec_pbl_m_list[0], dtype=float)), ' +/- ', np.std(np.array(spec_pbl_m_list[0], dtype=float)))
    print('\n')

    # saves percentage bone loss metrics for each class and average (All) as csv in args.save_loc
    mesial_pbl_metrics = pd.DataFrame({
        'Class': ['All'],
        'Precision_Mean': [np.nanmean(np.array(prec_pbl_m_list[0], dtype=float)).item()],
        'Precision_Std': [np.std(np.array(prec_pbl_m_list[0], dtype=float)).item()],
        'Recall_Mean': [np.nanmean(np.array(sens_pbl_m_list[0], dtype=float)).item()],
        'Recall_Std': [np.std(np.array(sens_pbl_m_list[0], dtype=float)).item()],
        'F1_Score_Mean': [np.nanmean(np.array(f1_pbl_m_list[0], dtype=float)).item()],
        'F1_Score_Std': [np.std(np.array(f1_pbl_m_list[0], dtype=float)).item()],
        'Specificity_Mean': [np.nanmean(np.array(spec_pbl_m_list[0], dtype=float)).item()],
        'Specificity_Std': [np.std(np.array(spec_pbl_m_list[0], dtype=float)).item()]
    })


    for i in range(1, len(sens_pbl_m_list)):
        count = i - 1 
        print('## '+clss_list[count]+' PBL ##')
        if sens_pbl_m_list[i] == []:
            sens_pbl_m_list[i] = [0.0]
        if spec_pbl_m_list[i] == []:
            spec_pbl_m_list[i] = [0.0]
        if prec_pbl_m_list[i] == []:
            prec_pbl_m_list[i] = [0.0]
        if f1_pbl_m_list[i] == []:
            f1_pbl_m_list[i] = [0.0]

        print('Precision: ', np.nanmean(np.array(prec_pbl_m_list[i], dtype=float)), ' +/- ', np.std(np.array(prec_pbl_m_list[i], dtype=float)))
        print('Recall: ', np.nanmean(np.array(sens_pbl_m_list[i], dtype=float)), ' +/- ', np.std(np.array(sens_pbl_m_list[i], dtype=float)))
        print('F1 Score: ', np.nanmean(np.array(f1_pbl_m_list[i], dtype=float)), ' +/- ', np.std(np.array(f1_pbl_m_list[i], dtype=float)))
        print('Specificity: ', np.nanmean(np.array(spec_pbl_m_list[i], dtype=float)), ' +/- ', np.std(np.array(spec_pbl_m_list[i], dtype=float)))
        print('\n')

        # save
        new_row = pd.DataFrame({
            'Class': [clss_list[count]],
            'Precision_Mean': [np.nanmean(np.array(prec_pbl_m_list[i], dtype=float)).item()],
            'Precision_Std': [np.std(np.array(prec_pbl_m_list[i], dtype=float)).item()],
            'Recall_Mean': [np.nanmean(np.array(sens_pbl_m_list[i], dtype=float)).item()],
            'Recall_Std': [np.std(np.array(sens_pbl_m_list[i], dtype=float)).item()],
            'F1_Score_Mean': [np.nanmean(np.array(f1_pbl_m_list[i], dtype=float)).item()],
            'F1_Score_Std': [np.std(np.array(f1_pbl_m_list[i], dtype=float)).item()],
            'Specificity_Mean': [np.nanmean(np.array(spec_pbl_m_list[i], dtype=float)).item()],
            'Specificity_Std': [np.std(np.array(spec_pbl_m_list[i], dtype=float)).item()]
        })
        mesial_pbl_metrics = pd.concat([mesial_pbl_metrics, new_row], ignore_index=True)
    mesial_pbl_metrics.to_csv(os.path.join(args.save_loc, 'mesial_pbl_metrics.csv'), index=False)



    print('###### Distal Bone Loss ######')
    print('## Average PBL ##')
    print('Precision: ', np.nanmean(np.array(prec_pbl_d_list[0], dtype=float)), ' +/- ', np.std(np.array(prec_pbl_d_list[0], dtype=float)))
    print('Recall: ', np.nanmean(np.array(sens_pbl_d_list[0], dtype=float)), ' +/- ', np.std(np.array(sens_pbl_d_list[0], dtype=float)))
    print('F1 Score: ', np.nanmean(np.array(f1_pbl_d_list[0], dtype=float)), ' +/- ', np.std(np.array(f1_pbl_d_list[0], dtype=float)))
    print('Specificity: ', np.nanmean(np.array(spec_pbl_d_list[0], dtype=float)), ' +/- ', np.std(np.array(spec_pbl_d_list[0], dtype=float)))
    print('\n')

    # saves
    distal_pbl_metrics = pd.DataFrame({
        'Class': ['All'],
        'Precision_Mean': [np.nanmean(np.array(prec_pbl_d_list[0], dtype=float)).item()],
        'Precision_Std': [np.std(np.array(prec_pbl_d_list[0], dtype=float)).item()],
        'Recall_Mean': [np.nanmean(np.array(sens_pbl_d_list[0], dtype=float)).item()],
        'Recall_Std': [np.std(np.array(sens_pbl_d_list[0], dtype=float)).item()],
        'F1_Score_Mean': [np.nanmean(np.array(f1_pbl_d_list[0], dtype=float)).item()],
        'F1_Score_Std': [np.std(np.array(f1_pbl_d_list[0], dtype=float)).item()],
        'Specificity_Mean': [np.nanmean(np.array(spec_pbl_d_list[0], dtype=float)).item()],
        'Specificity_Std': [np.std(np.array(spec_pbl_d_list[0], dtype=float)).item()]
    })


    for i in range(1, len(f1_pbl_d_list)):
        count = i - 1 
        print('## '+clss_list[count]+' PBL ##')
        if sens_pbl_d_list[i] == []:
            sens_pbl_d_list[i] = [0.0]
        if spec_pbl_d_list[i] == []:
            spec_pbl_d_list[i] = [0.0]
        if prec_pbl_d_list[i] == []:
            prec_pbl_d_list[i] = [0.0]
        if f1_pbl_d_list[i] == []:
            f1_pbl_d_list[i] = [0.0]

        print('Precision: ', np.nanmean(np.array(prec_pbl_d_list[i], dtype=float)), ' +/- ', np.std(np.array(prec_pbl_d_list[i], dtype=float)))
        print('Recall: ', np.nanmean(np.array(sens_pbl_d_list[i], dtype=float)), ' +/- ', np.std(np.array(sens_pbl_d_list[i], dtype=float)))
        print('F1 Score: ', np.nanmean(np.array(f1_pbl_d_list[i], dtype=float)), ' +/- ', np.std(np.array(f1_pbl_d_list[i], dtype=float)))
        print('Specificity: ', np.nanmean(np.array(spec_pbl_d_list[i], dtype=float)), ' +/- ', np.std(np.array(spec_pbl_d_list[i], dtype=float)))
        print('\n')

        # save
        new_row = pd.DataFrame({
            'Class': [clss_list[count]],
            'Precision_Mean': [np.nanmean(np.array(prec_pbl_d_list[i], dtype=float)).item()],
            'Precision_Std': [np.std(np.array(prec_pbl_d_list[i], dtype=float)).item()],
            'Recall_Mean': [np.nanmean(np.array(sens_pbl_d_list[i], dtype=float)).item()],
            'Recall_Std': [np.std(np.array(sens_pbl_d_list[i], dtype=float)).item()],
            'F1_Score_Mean': [np.nanmean(np.array(f1_pbl_d_list[i], dtype=float)).item()],
            'F1_Score_Std': [np.std(np.array(f1_pbl_d_list[i], dtype=float)).item()],
            'Specificity_Mean': [np.nanmean(np.array(spec_pbl_d_list[i], dtype=float)).item()],
            'Specificity_Std': [np.std(np.array(spec_pbl_d_list[i], dtype=float)).item()]
        })
        distal_pbl_metrics = pd.concat([distal_pbl_metrics, new_row], ignore_index=True)
    distal_pbl_metrics.to_csv(os.path.join(args.save_loc, 'distal_pbl_metrics.csv'), index=False)
    print('\n')


    # everything below is stored as [f0[average, c0, c1, ...] , f1[average,c2,c3, ...], ...]

    print('########## PRCK METRICS ##########')
    print('### Average Class PRCK ###')
    print('PRCK 0.5: ', np.nanmean(np.array(prck_thresh_list[0.5][0], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.5][0], dtype=float)))
    print('PRCK 0.25: ', np.nanmean(np.array(prck_thresh_list[0.25][0], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.25][0], dtype=float)))
    print('PRCK 0.05: ', np.nanmean(np.array(prck_thresh_list[0.05][0], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.05][0], dtype=float)))
    print('\n')
    print('\n')
    # saves prck metrics for each class as csv in args.save_loc
    prck_metrics = pd.DataFrame({
        'Class': ['All'],
        'PRCK_0.5_Mean': [np.nanmean(np.array(prck_thresh_list[0.5][0], dtype=float)).item()],
        'PRCK_0.5_Std': [np.std(np.array(prck_thresh_list[0.5][0], dtype=float)).item()],
        'PRCK_0.25_Mean': [np.nanmean(np.array(prck_thresh_list[0.25][0], dtype=float)).item()],
        'PRCK_0.25_Std': [np.std(np.array(prck_thresh_list[0.25][0], dtype=float)).item()],
        'PRCK_0.05_Mean': [np.nanmean(np.array(prck_thresh_list[0.05][0], dtype=float)).item()],
        'PRCK_0.05_Std': [np.std(np.array(prck_thresh_list[0.05][0], dtype=float)).item()]
    })
    # saves prck metric for all thresholds
    prck_thresh_all = pd.DataFrame({
        'Class': ['All'],
        '0.5': [np.nanmean(np.array(prck_thresh_list[0.5][0], dtype=float)).item()],
        '0.45': [np.nanmean(np.array(prck_thresh_list[0.45][0], dtype=float)).item()],
        '0.4': [np.nanmean(np.array(prck_thresh_list[0.4][0], dtype=float)).item()],
        '0.35': [np.nanmean(np.array(prck_thresh_list[0.35][0], dtype=float)).item()],
        '0.3': [np.nanmean(np.array(prck_thresh_list[0.3][0], dtype=float)).item()],
        '0.25': [np.nanmean(np.array(prck_thresh_list[0.25][0], dtype=float)).item()],
        '0.2': [np.nanmean(np.array(prck_thresh_list[0.2][0], dtype=float)).item()],
        '0.15': [np.nanmean(np.array(prck_thresh_list[0.15][0], dtype=float)).item()],
        '0.1': [np.nanmean(np.array(prck_thresh_list[0.1][0], dtype=float)).item()],
        '0.05': [np.nanmean(np.array(prck_thresh_list[0.05][0], dtype=float)).item()]

    })


    for i in range(1, len(prck_thresh_list[0.5])):
        count = i - 1 
        print('### C'+str(count)+' Class PRCK ###')
        print('PRCK 0.5: ', np.nanmean(np.array(prck_thresh_list[0.5][i], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.5][i], dtype=float)))
        print('PRCK 0.25: ', np.nanmean(np.array(prck_thresh_list[0.25][i], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.25][i], dtype=float)))
        print('PRCK 0.05: ', np.nanmean(np.array(prck_thresh_list[0.05][i], dtype=float)), ' +/- ', np.std(np.array(prck_thresh_list[0.05][i], dtype=float)))
        print('\n')

        # saves
        new_row = pd.DataFrame({
            'Class': ['C'+str(count)],
            'PRCK_0.5_Mean': [np.nanmean(np.array(prck_thresh_list[0.5][i], dtype=float)).item()],
            'PRCK_0.5_Std': [np.std(np.array(prck_thresh_list[0.5][i], dtype=float)).item()],
            'PRCK_0.25_Mean': [np.nanmean(np.array(prck_thresh_list[0.25][i], dtype=float)).item()],
            'PRCK_0.25_Std': [np.std(np.array(prck_thresh_list[0.25][i], dtype=float)).item()],
            'PRCK_0.05_Mean': [np.nanmean(np.array(prck_thresh_list[0.05][i], dtype=float)).item()],
            'PRCK_0.05_Std': [np.std(np.array(prck_thresh_list[0.05][i], dtype=float)).item()]
        })
        prck_metrics = pd.concat([prck_metrics, new_row], ignore_index=True)
        new_row_thresh = pd.DataFrame({
        'Class': ['C'+str(count)],
        '0.5': [np.nanmean(np.array(prck_thresh_list[0.5][i], dtype=float)).item()],
        '0.45': [np.nanmean(np.array(prck_thresh_list[0.45][i], dtype=float)).item()],
        '0.4': [np.nanmean(np.array(prck_thresh_list[0.4][i], dtype=float)).item()],
        '0.35': [np.nanmean(np.array(prck_thresh_list[0.35][i], dtype=float)).item()],
        '0.3': [np.nanmean(np.array(prck_thresh_list[0.3][i], dtype=float)).item()],
        '0.25': [np.nanmean(np.array(prck_thresh_list[0.25][i], dtype=float)).item()],
        '0.2': [np.nanmean(np.array(prck_thresh_list[0.2][i], dtype=float)).item()],
        '0.15': [np.nanmean(np.array(prck_thresh_list[0.15][i], dtype=float)).item()],
        '0.1': [np.nanmean(np.array(prck_thresh_list[0.1][i], dtype=float)).item()],
        '0.05': [np.nanmean(np.array(prck_thresh_list[0.05][i], dtype=float)).item()]
        })
        prck_thresh_all = pd.concat([prck_thresh_all, new_row_thresh], ignore_index=True)
        
    prck_metrics.to_csv(os.path.join(args.save_loc, 'prck_metrics.csv'), index=False)
    prck_thresh_all.to_csv(os.path.join(args.save_loc, 'prck_thresh_all.csv'), index=False)


    print('########## NME METRICS ##########')
    print('### Average Class NME ###')
    print(np.nanmean(np.array(nme_list[0], dtype=float)), ' +/- ', np.std(np.array(nme_list[0], dtype=float)))
    print('\n')

    # saves nme metrics for each class as csv in args.save_loc
    nme_metrics = pd.DataFrame({
        'Class': ['All'],
        'NME_Mean': [np.nanmean(np.array(nme_list[0], dtype=float)).item()],
        'NME_Std': [np.std(np.array(nme_list[0], dtype=float)).item()]
    })

    for nme_itm in range(1, len(nme_list)):
        count = nme_itm - 1 
        print('### C', str(count), ' Class NME ###')
        print(np.nanmean(np.array(nme_list[nme_itm], dtype=float)), ' +/- ', np.std(np.array(nme_list[nme_itm], dtype=float)))
        print('\n')
        # saves
        new_row = pd.DataFrame({
            'Class': ['C'+str(count)],
            'NME_Mean': [np.nanmean(np.array(nme_list[nme_itm], dtype=float)).item()],
            'NME_Std': [np.std(np.array(nme_list[nme_itm], dtype=float)).item()]
        })
        nme_metrics = pd.concat([nme_metrics, new_row], ignore_index=True)
    nme_metrics.to_csv(os.path.join(args.save_loc, 'nme_metrics.csv'), index=False)