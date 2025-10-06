import numpy as np
import cv2
import matplotlib.pyplot as plt



def wrap180_deg(a):
    """
    Wrap any angle (deg) into [-90, 90].
    Vectorized, preserves shape.
    """
    a = (a + 180.0) % 360.0 - 180.0
    a = np.where(a < -90.0, a + 180.0, a)
    a = np.where(a >  90.0, a - 180.0, a)
    return a

def nmse_range_angles(target, pred):
    """
    Normalized MSE over wrapped angular errors (degrees).
    NMSE_range = mean((Δ/90)^2), with Δ = wrap180(pred - target).

    Returns np.nan if either array is empty after masking.
    """
    target = np.asarray(target, dtype=float)
    pred   = np.asarray(pred, dtype=float)

    if target.size == 0 or pred.size == 0:
        return np.nan

    
    mask = np.isfinite(target) & np.isfinite(pred)
    if not np.any(mask):
        return np.nan

    delta = wrap180_deg(pred[mask] - target[mask])
    return float(np.mean((delta / 90.0) ** 2))




def get_nmse(target, pred, clss):
    """
    Calculates NMSE for each object for a single image, per class.

    Returns a list of NMSE values per object, ordered per class.
    """
    nmse_list_average, nmse_list_class = [], []
    # for each input traverse clss and places the items into a dictonary with the class as the key
    for fold, item in enumerate(clss):
        nmse_list_class.append([])
        # creates dictonary for first 3 classes (tooth boxes)
        target_dict = {0: [], 1: [], 2: []}
        pred_dict = {0: [], 1: [], 2: []}

        # flattens lists
        item = [i for sublist in item for i in sublist]
        target_fold = [i for sublist in target[fold] for i in sublist]
        pred_fold = [i for sublist in pred[fold] for i in sublist]
        for i, c in enumerate(item):
            if c in target_dict.keys():
                target_dict[c].append(target_fold[i])
                pred_dict[c].append(pred_fold[i])

        # flattens target and pred lists
        avg_target_eval_clss, avg_pred_eval_clss = [], []
        for eval_clss in target_dict.keys():
            # combines eval_clss list 
            avg_target_eval_clss += target_dict[eval_clss]
            avg_pred_eval_clss += pred_dict[eval_clss]

            target_cur = np.array(target_dict[eval_clss], dtype=float)
            pred_cur = np.array(pred_dict[eval_clss], dtype=float)

            nmse_list_class[fold].append(nmse_range_angles(target_cur, pred_cur))
    return nmse_list_class 


def calculate_metrics(tp,fp,tn,fn):
    """
    Calculates sensitivity, specificity, precision and f1 score from confusion matrix values.

    Returns sensitivity, specificity, precision and f1 values, from tp, fp, tn, fn input values.
    """

    if tp == 0:
        sensitivity = 0
        precision = 0
        f1 = 0
    else:
        # sensitivity or recall
        sensitivity = tp / (tp + fn)
        # precision or ppv
        precision = tp / (tp + fp)
        # f1 score
        f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
    # specificity
    if tn == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    return sensitivity, specificity, precision, f1


def calculate_furcation_metrics(lbl_fa, lbl_fblm, lbl_fbld, pred_fa, pred_fblm, pred_fbld, average_box_width, average_box_height, tp, fp, tn, fn, furcation_dist_thresh=0.05):
    """
    Calculates furcation involvement confusion matrix values from target and pred furcation apex and furcation bone level points.

    Uses a furcation_distance_thresh and average box diagonal distance to determine the distance required to classify a furcation as healthy or involved (0.05).
    Adds value to appropriate tp, fp, tn, fn, counters, for a single object/site. 

    Returns tp, fp, tn, fn values for metric
    """
    
    # if the distance from FLB-m to FA and FLB-d to FA is less than the threshold of the average diagonal distance of all bounding boxes in the image then the furcation is considered healthy
    # calculates the diagonal distance of the target box
    target_diag = np.sqrt(average_box_height**2 + average_box_width**2)
    # finds the longest distance between the furcation bone level mesial and furcation apex for target
    target_fa_dist = max([np.linalg.norm(np.array(lbl_fblm) - np.array(lbl_fa)), np.linalg.norm(np.array(lbl_fbld) - np.array(lbl_fa))])
    # finds the longest distance between the furcation bone level distal and furcation apex for pred
    pred_fa_dist = max([np.linalg.norm(np.array(pred_fblm) - np.array(pred_fa)), np.linalg.norm(np.array(pred_fbld) - np.array(pred_fa))])

    # finds the target furcation involvement binary classification. if target_fa_dist is 0 then the furcation is healthy, else it is involved (less than 0.01 to allow for small errors in labelling)
    if target_fa_dist <= 0.90:
        target_furcation_clss = 0
    else:
        target_furcation_clss = 1
    # finds the pred furcation involvement binary classification. if pred_fa_dist is less than the threshold of the average diagonal distance of all bounding boxes in the image then the furcation is healthy, else it is involved
    if pred_fa_dist < target_diag * furcation_dist_thresh:
        pred_furcation_clss = 0
    else:
        pred_furcation_clss = 1

    # calculates conf matrix for positive and negative classes
    for i in range(2):
        # calculates the confusion matrix for the furcation involvement classification
        if target_furcation_clss == i and pred_furcation_clss == i:
            tp[i] += 1
        if target_furcation_clss != i and pred_furcation_clss == i:
            fp[i] += 1
        if target_furcation_clss == i and pred_furcation_clss != i:
            fn[i] += 1
        if target_furcation_clss != i and pred_furcation_clss != i:
            tn[i] += 1

    return tp, fp, tn, fn



# gets the confusion matrix for the target and pred bone loss. classifies the percentage of bone loss as either Healthy (<15), Early (15-33), Moderate (33-66) or severe (>66)
# target: target bone loss percentage
# pred: predicted bone loss percentage
# tp: true positive count for each bone loss classification [Healthy, Early, Moderate, Severe]
# fp: false positive count for each bone loss classification [Healthy, Early, Moderate, Severe]
# tn: true negative count for each bone loss classification [Healthy, Early, Moderate, Severe]
# fn: false negative count for each bone loss classification [Healthy, Early, Moderate, Severe]
# class_counts: counts the target instances for each bone loss classification [Healthy, Early, Moderate, Severe]
def add_to_metrics(target, pred, tp, fp, tn, fn, class_counts=[0,0,0,0]):
    """
    Calculates percentage bone loss confusion matrix values from pre-calculated target and pred bone loss percentages.

    classifies percentage of bone loss to Healthy (<15%), Early (15-33%), Moderate (33-66%) or Severe (>66%).
    Adds value to appropriate tp, fp, tn, fn counters, for a single object/site.

    Returns tp, fp, tn, fn values
    """

    # classifies the percentage of bone loss
    # target
    if target < 0.15:
        bl_class_lbl = 0
        class_counts[0] += 1
    elif target >= 0.15 and target < 0.33:
        bl_class_lbl = 1
        class_counts[1] += 1
    elif target >= 0.33 and target < 0.66:
        bl_class_lbl = 2
        class_counts[2] += 1
    elif target >= 0.66:
        bl_class_lbl = 3
        class_counts[3] += 1
    # pred
    if pred < 0.15:
        bl_class_pred = 0
    elif pred >= 0.15 and pred < 0.33:
        bl_class_pred = 1
    elif pred >= 0.33 and pred < 0.66:
        bl_class_pred = 2
    elif pred >= 0.66:
        bl_class_pred = 3

    # multi-class disease matrix
    # loops all classes
    for i in range(4):
        # tp 
        if bl_class_lbl == i and bl_class_pred == i:
            tp[i] += 1
        if bl_class_lbl != i and bl_class_pred == i:
            fp[i] += 1
        if bl_class_lbl == i and bl_class_pred != i:
            fn[i] += 1
        if bl_class_lbl != i and bl_class_pred != i:
            tn[i] += 1

    return tp, fp, tn, fn


def find_intersection(p1, p2, theta):
    """
    Finds the intersection point of a line rotated by theta degrees from the vertical line through point p1, and the line through points p1 and p2.

    Returns the intersection point
    """
    # Create the direction vector for the vertical line through p1
    vertical_direction = np.array([0, 1])  # Vertical line (0, 1)

    # Rotate the direction vector by theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta),  np.cos(theta)]])
    
    # New direction after rotation
    rotated_direction = rotation_matrix @ vertical_direction

    # Find the normal vector for the rotated line (perpendicular vector)
    normal_vector = np.array([-rotated_direction[1], rotated_direction[0]])

    # Define the system of equations
    A = np.array([rotated_direction, -normal_vector]).T
    b = p2 - p1

    # Solve for t and s
    t_s = np.linalg.solve(A, b)

    # Find the intersection point on the rotated line
    intersection_point = p1 + t_s[0] * rotated_direction
    
    return intersection_point # , rotated_direction


def calculate_pbl(cej, bl, rl):
    """
    Calculates the percentage of bone loss for a given RL root level point and BL bone level point. pbl is cej to bl distance / cej to rl distance for this paper.


    Returns percent bone loss value (0.0 - 1.0) for the given RL point.
    """
    # calculates the distance between the cej and bl points
    cej_bl_dist = np.linalg.norm(np.array(cej) - np.array(bl))
    # calculates the distance between the cej and rl points
    cej_rl_dist = np.linalg.norm(np.array(cej) - np.array(rl))
    # calculates the percentage of bone loss
    pbl = (cej_bl_dist / cej_rl_dist)
    return pbl


def calculate_bone_loss(cej, bl, rl_1, rl_2, rl_3, rotate, img_object):
    """
    Prepares the to calculate percent of bone loss (PBL) for a single tooth.
    Aligns bl and rl points to the rotated tooth axis, using the cej as the rotation point.
    PBL is calculated for each root for the current tooth object.

    Returns a list of percentage of bone loss values (0.0 - 1.0) for each root in the tooth object. Also returns the image with the points drawn on it for visual verification.
    """

    # converts rotate to radians
    rotate = np.radians(rotate)


    img_object = cv2.circle(img_object, (int(cej[0]), int(cej[1])), 5, (0,255,0), -1)
    img_object = cv2.circle(img_object, (int(bl[0]), int(bl[1])), 5, (0,255,0), -1)


    # finds rotated intersection point
    bl = find_intersection(cej, bl, rotate)
    img_object = cv2.circle(img_object, (int(bl[0]), int(bl[1])), 5, (255,0,0), -1)

    rl_pbl = [] # store PBL for percentage of bone loss for each RL point, bone loss is measured from the shortest root, so highest pbl value is the correct one to use
    if not np.isnan(rl_1[0]) or not np.isnan(rl_1[1]):
        img_object = cv2.circle(img_object, (int(rl_1[0]), int(rl_1[1])), 5, (0,255,0), -1)
        rl_1 = find_intersection(cej, rl_1, rotate)
        img_object = cv2.circle(img_object, (int(rl_1[0]), int(rl_1[1])), 5, (255,0,0), -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_1))
    else:
        rl_pbl.append(-1.0)
    if not np.isnan(rl_2[0]) or not np.isnan(rl_2[1]):
        img_object = cv2.circle(img_object, (int(rl_2[0]), int(rl_2[1])), 5, (0,255,0), -1)
        rl_2 = find_intersection(cej, rl_2, rotate)
        img_object = cv2.circle(img_object, (int(rl_2[0]), int(rl_2[1])), 5, (255,0,0), -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_2))
    else:
        rl_pbl.append(-1.0)
    if not np.isnan(rl_3[0]) or not np.isnan(rl_3[1]):
        img_object = cv2.circle(img_object, (int(rl_3[0]), int(rl_3[1])), 5, (0,255,0), -1)
        rl_3 = find_intersection(cej, rl_3, rotate)
        img_object = cv2.circle(img_object, (int(rl_3[0]), int(rl_3[1])), 5, (255,0,0), -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_3))
    else:
        rl_pbl.append(-1.0)

    return rl_pbl, img_object


def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Returns the IoU value (0.0 - 1.0) for the given boxes.
    """
    # copy box2 to prevent changing the original box2
    box2 = box2.copy()
    box1 = box1.copy()

    # converts box coordinates from x_top_left, y_top_left, w, h, to x_top_left, y_top_left, x_bottom_right, y_bottom_right
    box1[2] = box1[0] + box1[2]
    box1[3] = box1[1] + box1[3]
    box2[2] = box2[0] + box2[2]
    box2[3] = box2[1] + box2[3]

    # finds the intersection of the two boxes
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # finds the area of the intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # finds the area of the two boxes

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # calculates the union of the two boxes
    unionArea = box1Area + box2Area - interArea

    # calculates the intersection over union
    iou = interArea / unionArea

    return iou



def prck(primary, secondary, avg_target_box_height, avg_target_box_width, num, denom, threshold, include_non_matched=False):
    """
    Calculates the Precentage of Relative Correct Keypoints (PRCK) at a specified threshold for each keypoint class in a single tooth/bounding box object.
    Adds value to the numerator and denominator for each pred and target keypoint for each keypoint class.
    
    Returns numerator and denominator for PRCK calculation for each keypoint in an image.
    """
    #extracts values
    target_list, pred_list, visibility_list = primary[0], primary[1], primary[2]
    remaining_target_kpts, remaining_pred_kpts, remaining_target_vis, remaining_pred_vis = secondary[0], secondary[1], secondary[2], secondary[3]
    # finds the diagonal distance of the target box
    target_diag = np.sqrt(avg_target_box_height**2 + avg_target_box_width**2)
    for p_item in range(len(target_list)):
        target, pred, visibility = target_list[p_item], pred_list[p_item], visibility_list[p_item]
        # loops through each keypoint, if the keypoint is a visiblity != 0 then calculate pdj
        for i in range(len(target)):
            if visibility[i] != 0:
                # calculate the euclidean distance between two x, y points (target and pred keypoint)
                dist = np.linalg.norm(np.array(target[i]) - np.array(pred[i]))
                # if the distance is less than 0.5 then add 1 to the count of detected joints
                if dist < target_diag * threshold:
                    num[i] += 1
                    denom[i] += 1
                else:
                    denom[i] += 1

    if include_non_matched:
        # handels false positives and false negatives
        if remaining_target_kpts != []:
            for r_item1 in range(len(remaining_target_kpts)):
                if remaining_target_vis[r_item1][0] != 0:
                    denom[r_item1] += 1

        if remaining_pred_kpts != []:
            for r_item2 in range(len(remaining_pred_kpts)):
                if remaining_pred_vis[r_item2][0] != 0:
                    denom[r_item2] += 1

    return num, denom




def nme(primary, secondary, avg_target_box_height, avg_target_box_width, norm_error, include_non_matched=False):
    """
    Calculates the Normalised Mean Error (NME) for keypoints for a single matched object.

    Returns the NME value (0.0 - 1.0) for the given class.
    """
    # finds the diagonal distance of the target box (average normalising factor)
    target_diag = np.sqrt(avg_target_box_height**2 + avg_target_box_width**2)
    # extracts values
    target, pred, visibility = primary[0], primary[1], primary[2]
    remaining_target, remaining_pred, remaining_target_vis, remaining_pred_vis = secondary[0], secondary[1], secondary[2], secondary[3]
    # calculates normalisd mean error for each keypoint
    for item in range(len(target)):
        target_list, pred_list, visibility_list = target[item], pred[item], visibility[item]

        # convert visibility list into 0 and 1s where 0 is 0 and 1 or 2 is 1
        visibility_list = [1 if v != 0 else 0 for v in visibility_list]

        # converts to np array
        target_list = np.array(target_list)
        pred_list = np.array(pred_list)
        visibility_list = np.array(visibility_list)

        # calcukates the l2 norm of the target and pred keypoint
        error = np.linalg.norm(target_list - pred_list, axis=1)

        # normalise the distance by the target diagonal distance
        normal_error = error / target_diag

        # appends the normalised error to the norm_error at each relative list position
        for i in range(len(normal_error)):
            if visibility_list[i] != 0:
                norm_error[i].append(normal_error[i])

    if include_non_matched:
        # handels false positives and false negatives
        if remaining_target != []:
            for r_item1 in range(len(remaining_target)):
                if remaining_target_vis[r_item1][0] != 0:
                    norm_error[r_item1].append(1.0)
        if remaining_pred != []:
            for r_item2 in range(len(remaining_pred)):
                if remaining_pred_vis[r_item2][0] != 0:
                    norm_error[r_item2].append(1.0)
    # print(norm_error)
    return norm_error