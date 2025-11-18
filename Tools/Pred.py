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
from All_Metrics import find_intersection, calculate_pbl




VISUAL_CFG = {
    # BGR colours
    # Bounding boxes
    "box_colour": (255, 20, 20),        # BGR
    "box_thickness": 3,
    "box_text_colour": (255, 255, 255),
    "box_text_bg": (255, 0, 0),       # filled rectangle behind text
    "box_text_scale": 0.8,
    "box_text_thickness": 2,

    # Keypoints
    "kpt_colour": (255, 20, 20),
    "kpt_radius": 10,
    "kpt_thickness": -1,              # filled circle

    # PBL lines
    "pbl_line_colour": (0, 255, 255),      # default/fallback colour (BGR)
    "pbl_line_thickness": 3,
    "pbl_dashed_colour": (200, 200, 200),
    "pbl_dashed_thickness": 3,
    "pbl_marker_colour": (0, 0, 255),
    "pbl_marker_radius": 6,

    # PBL text (default)
    "pbl_text_colour": (0, 0, 255),
    "pbl_text_scale": 1.0,
    "pbl_text_thickness": 2,
    "pbl_text_bg": (255, 255, 255),   # optional background (set None to disable)
    "pbl_text_bg_opacity": 0.8,

    # Severity colour map (BGR) for solid lines
    "pbl_colour_map": {
        "Healthy": (0, 255, 0),
        "Mild":    (0, 255, 255),
        "Moderate":(0, 165, 255),
        "Severe":  (0, 0, 255),
        "Unknown": (200, 200, 200)
    },

    # Severity colour map (BGR) for classification text
    # Slightly darker tones for readability against white bg
    # "pbl_text_colour_map": {
    #     "Healthy": (0, 180, 0),
    #     "Mild":    (0, 180, 180),
    #     "Moderate":(0, 120, 200),
    #     "Severe":  (0, 0, 180),
    #     "Unknown": (90, 90, 90)
    # },

    # Furcation (FA) line + text colours
    "fa_colour_healthy": (0, 255, 0),
    "fa_colour_involved": (0, 0, 255),
    "fa_text_colour_map": {
        "Healthy":  (0, 255, 0),
        "Involved": (0, 0, 255)
    },

    # Apply severity to TEXT BACKGROUND instead of text colour for PBL
    # If not provided, code will fall back to pbl_colour_map for background.
    "pbl_text_bg_map": {
        "Healthy": (0, 100, 0),
        "Mild":    (0, 100, 100),
        "Moderate":(0, 50, 100),
        "Severe":  (0, 0, 100),
        "Unknown": (200, 200, 200)
    },
    # Optional per-severity TEXT colour overrides for PBL labels
    "pbl_text_fg_map": {
        "Healthy":  (255, 255, 255),
        "Mild":     (255, 255, 255),
        "Moderate": (255, 255, 255),
        "Severe":   (255, 255, 255),
        "Unknown":  (0, 0, 0)
    },

    # Furcation text background map + optional opacity override
    "fa_text_bg_map": {
        "Healthy":  (0, 100, 0),
        "Involved": (0, 0, 100)
    },
    "fa_text_bg_opacity": 0.7,
    # Optional per-class text colour overrides for FA labels (falls back to existing map)
    "fa_text_fg_map": {
        "Healthy":  (255, 255, 255),
        "Involved": (255, 255, 255)
    }
}

# removes none and nan values from results list
def filter_none(lst, replace_val = 0.0):
    if isinstance(lst[0], list):
        return [[replace_val if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in row] for row in lst]
    else:
        return [replace_val if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in lst]


def _pbl_severity_class(pbl_value):
    if pbl_value is None or (isinstance(pbl_value, float) and math.isnan(pbl_value)):
        return "Unknown"
    if pbl_value < 0.15:
        return "Healthy" # (<0.15)
    if pbl_value < 0.33:
        return "Mild" # (0.15-0.33)
    if pbl_value < 0.66:
        return "Moderate" # (0.33-0.66)
    return "Severe" # (>0.66)


# --- helper: put text with optional filled bg (alpha blended) ---
def _put_text_with_bg(img, text, org, colour, scale, thickness, bg_colour=None, bg_opacity=0.6):
    if not text:
        return img
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    x = max(0, x); y = max(th + 2, y)  # ensure visible
    if bg_colour is not None:
        # rectangle coords
        x2, y2 = x + tw + 6, y + 4
        x1, y1 = x - 3, y - th - 2
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
        cv2.addWeighted(overlay, bg_opacity, img, 1 - bg_opacity, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
    return img


# --- helper: dashed line from pt1->pt2 ---
def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=8, gap_len=6):
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    vec = pt2 - pt1
    dist = np.linalg.norm(vec)
    if dist == 0:
        return img
    direction = vec / dist
    cur = pt1.copy()
    while np.linalg.norm(cur - pt1) < dist:
        end = cur + direction * dash_len
        if np.linalg.norm(end - pt1) > dist:
            end = pt2
        cv2.line(img, (int(cur[0]), int(cur[1])), (int(end[0]), int(end[1])), color, thickness)
        cur = end + direction * gap_len
    return img


# ---- Inline calculate_bone_loss (with 0/0.0 filtering) ----
def calculate_bone_loss(cej, bl, rl_1, rl_2, rl_3, rotate, img_object):

    rotate = np.radians(rotate)

    img_object = cv2.circle(img_object, (int(cej[0]), int(cej[1])), VISUAL_CFG["kpt_radius"], (0,255,0), VISUAL_CFG["kpt_thickness"])
    img_object = cv2.circle(img_object, (int(bl[0]), int(bl[1])), VISUAL_CFG["kpt_radius"], (0,255,0), VISUAL_CFG["kpt_thickness"])

    # rotate/intersect BL
    bl = find_intersection(cej, bl, rotate)
    img_object = cv2.circle(img_object, (int(bl[0]), int(bl[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)

    rl_pbl = []
    if not (np.isnan(rl_1[0]) or np.isnan(rl_1[1]) or rl_1[0] == 0 or rl_1[1] == 0 or rl_1[0] == 0.0 or rl_1[1] == 0.0):
        img_object = cv2.circle(img_object, (int(rl_1[0]), int(rl_1[1])), VISUAL_CFG["kpt_radius"], (0,255,0), VISUAL_CFG["kpt_thickness"])
        rl_1 = find_intersection(cej, rl_1, rotate)
        img_object = cv2.circle(img_object, (int(rl_1[0]), int(rl_1[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_1))
    else:
        rl_pbl.append(-1.0)

    if not (np.isnan(rl_2[0]) or np.isnan(rl_2[1]) or rl_2[0] == 0 or rl_2[1] == 0 or rl_2[0] == 0.0 or rl_2[1] == 0.0):
        img_object = cv2.circle(img_object, (int(rl_2[0]), int(rl_2[1])), VISUAL_CFG["kpt_radius"], (0,255,0), VISUAL_CFG["kpt_thickness"])
        rl_2 = find_intersection(cej, rl_2, rotate)
        img_object = cv2.circle(img_object, (int(rl_2[0]), int(rl_2[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_2))
    else:
        rl_pbl.append(-1.0)

    if not (np.isnan(rl_3[0]) or np.isnan(rl_3[1]) or rl_3[0] == 0 or rl_3[1] == 0 or rl_3[0] == 0.0 or rl_3[1] == 0.0):
        img_object = cv2.circle(img_object, (int(rl_3[0]), int(rl_3[1])), VISUAL_CFG["kpt_radius"], (0,255,0), VISUAL_CFG["kpt_thickness"])
        rl_3 = find_intersection(cej, rl_3, rotate)
        img_object = cv2.circle(img_object, (int(rl_3[0]), int(rl_3[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)
        rl_pbl.append(calculate_pbl(cej, bl, rl_3))
    else:
        rl_pbl.append(-1.0)

    return rl_pbl, img_object


# --- helper: compute text rectangle (matching _put_text_with_bg layout) ---
def _text_rect_for_org(text, org, scale, thickness):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    x = max(0, x); y = max(th + 2, y)
    x2, y2 = x + tw + 6, y + 4
    x1, y1 = x - 3, y - th - 2
    return (x1, y1, x2, y2), (tw, th)

def _rects_overlap(a, b, pad=0):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= pad; ay1 -= pad; ax2 += pad; ay2 += pad
    bx1 -= pad; by1 -= pad; bx2 += pad; by2 += pad
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

# --- helper: nudge label to nearest free spot so it does not overlap other labels ---
def _find_non_overlapping_label_pos(preferred_org, text, img_shape, scale, thickness, occupied_rects, margin=2):
    h, w = img_shape[:2]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    steps = [0, 10, 20, 30, 45, 60, 80, 100]
    dirs = [(1,-1), (1,1), (-1,-1), (-1,1), (0,-1), (0,1), (1,0), (-1,0)]
    for r in steps:
        for dxs, dys in dirs:
            cand = (preferred_org[0] + dxs * r, preferred_org[1] + dys * r)
            cx, cy = cand
            cx = max(0, min(w - tw - 6, cx))
            cy = max(th + 2, min(h - 4, cy))
            rect, _ = _text_rect_for_org(text, (cx, cy), scale, thickness)
            if rect[0] < 0 or rect[1] < 0 or rect[2] > w or rect[3] > h:
                continue
            if all(not _rects_overlap(rect, r2, pad=margin) for r2 in occupied_rects):
                occupied_rects.append(rect)
                return (cx, cy)
    rect, _ = _text_rect_for_org(text, preferred_org, scale, thickness)
    occupied_label_rects.append(rect)
    return preferred_org

# --- helper: furcation involvement classification (pixels) ---
def _classify_furcation(pred_fa, pred_fbl_m, pred_fbl_d, avg_box_w, ratio_thresh=0.05):
    pts = []
    if pred_fa is None or np.any(np.isnan(pred_fa)):
        return None, None
    if pred_fbl_m is not None and not np.any(np.isnan(pred_fbl_m)):
        pts.append(np.linalg.norm(np.array(pred_fa) - np.array(pred_fbl_m)))
    if pred_fbl_d is not None and not np.any(np.isnan(pred_fbl_d)):
        pts.append(np.linalg.norm(np.array(pred_fa) - np.array(pred_fbl_d)))
    if len(pts) == 0 or avg_box_w is None or avg_box_w <= 0:
        return None, None
    max_dist = float(max(pts))
    label = "Healthy" if max_dist < (ratio_thresh * float(avg_box_w)) else "Involved"
    return label, max_dist

# --- helper: extract pixel keypoints and validity (handles [x,y] or [x,y,vis]) ---
def _extract_kpts_pix_and_valid(kpts_row, img_w, img_h):
    kpts_row = np.asarray(kpts_row)
    has_vis = (kpts_row.shape[-1] >= 3)
    pix = []
    valid = []
    for kp in kpts_row:
        if has_vis:
            x_n, y_n, v = float(kp[0]), float(kp[1]), float(kp[2])
        else:
            x_n, y_n, v = float(kp[0]), float(kp[1]), 1.0
        is_valid = (v > 0.0) and (not math.isnan(x_n)) and (not math.isnan(y_n)) and (x_n != 0.0) and (y_n != 0.0)
        valid.append(is_valid)
        pix.append([x_n * img_w, y_n * img_h])
    return np.array(pix, dtype=float), np.array(valid, dtype=bool)


def Pred_Model(weight_path_orig, pred_imgs_orig, best_seg_weights, ignore_box_classes, box_classes, args):
    """
    Single-weight pipeline with visual styling controlled by VISUAL_CFG:
      • Predict boxes+keypoints
      • Save YOLO-format TXT predictions (normalised)
      • Draw keypoints, PBL lines/labels (always)
      • Draw boxes unless their class is in ignore_box_classes (if provided)
    """

    saved_images = []

    model_kpts = YOLO(weight_path_orig)
    model_seg  = YOLO(best_seg_weights)

    vis_root = os.path.join(args.save_loc, "visualisations")
    pred_root = os.path.join(args.save_loc, "predictions")
    os.makedirs(vis_root, exist_ok=True)
    os.makedirs(pred_root, exist_ok=True)

    ignore_set = None
    if ignore_box_classes is not None:
        try:
            ignore_set = set(int(x) for x in ignore_box_classes)
        except Exception:
            ignore_set = None

    for img in os.listdir(pred_imgs_orig):
        if not img.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
            continue

        image_path = os.path.join(pred_imgs_orig, img)
        img_object = cv2.imread(image_path)
        if img_object is None:
            continue

        img_h, img_w = img_object.shape[:2]

        pred = model_kpts.predict(
            image_path,
            save=False,
            imgsz=args.image_size,
            iou=args.pred_kpts_iou,
            conf=args.pred_kpts_conf,
        )

        # Rotation + (optional) post-processed kpts (normalised space)
        if args.post_process_kpts:
            new_pred, old_kpts, _, _, _, rotate_pred = post_process(model_seg, pred, image_path, args)
            pred_kpt_xyn = torch.tensor(new_pred, device=pred[0].keypoints.xyn.device)
        else:
            _, _, _, _, _, rotate_pred = post_process(model_seg, pred, image_path, args, img_dims_only=True)
            pred_kpt_xyn = pred[0].keypoints.xyn

        # pull predictions (normalised)
        box_xywhn = pred[0].boxes.xywhn
        box_cls   = pred[0].boxes.cls
        box_conf  = pred[0].boxes.conf

        box_xywhn = box_xywhn.detach().cpu().numpy() if hasattr(box_xywhn, "detach") else np.array(box_xywhn)
        box_cls   = box_cls.detach().cpu().numpy().astype(int) if hasattr(box_cls, "detach") else np.array(box_cls).astype(int)
        box_conf  = box_conf.detach().cpu().numpy().astype(float) if hasattr(box_conf, "detach") else np.array(box_conf).astype(float)
        kpts_xyn  = pred_kpt_xyn.detach().cpu().numpy() if hasattr(pred_kpt_xyn, "detach") else np.array(pred_kpt_xyn)

        img_vis = copy.deepcopy(img_object)

        # keep track of placed text rectangles so labels don't overlap
        occupied_label_rects = []

        # compute average box width in pixels for this image (for furcation threshold)
        avg_box_w = None
        if box_xywhn.size > 0:
            widths_pix = box_xywhn[:, 2] * img_w
            valid_w = widths_pix[widths_pix > 0]
            if valid_w.size > 0:
                avg_box_w = float(np.mean(valid_w))

        # TXT save
        txt_path = os.path.join(pred_root, img.rsplit(".", 1)[0] + ".txt")
        with open(txt_path, "w") as f:
            for i in range(len(box_xywhn)):
                cx_n, cy_n, w_n, h_n = box_xywhn[i]
                cls_idx = int(box_cls[i]) if i < len(box_cls) else 0
                conf    = float(box_conf[i]) if i < len(box_conf) else 0.0
                cls_name = (box_classes[cls_idx] if (box_classes is not None and cls_idx < len(box_classes))
                            else f"C{cls_idx}")

                # pixel box corners
                x_n = cx_n - w_n/2.0
                y_n = cy_n - h_n/2.0
                x1, y1, w_pix, h_pix = int(x_n*img_w), int(y_n*img_h), int(w_n*img_w), int(h_n*img_h)
                p1, p2 = (x1, y1), (x1+w_pix, y1+h_pix)

                # --- draw box unless ignored ---
                draw_box = not (ignore_set is not None and cls_idx in ignore_set)
                if draw_box:
                    cv2.rectangle(img_vis, p1, p2, VISUAL_CFG["box_colour"], VISUAL_CFG["box_thickness"])
                    # reserve box label area as well (helps reduce conflicts)
                    box_label = f"{cls_name}" # {conf:.2f}
                    box_org = (p1[0], max(0, p1[1]-8))
                    rect, _ = _text_rect_for_org(box_label, box_org, VISUAL_CFG["box_text_scale"], VISUAL_CFG["box_text_thickness"])
                    occupied_label_rects.append(rect)
                    _put_text_with_bg(
                        img_vis,
                        box_label,
                        box_org,
                        VISUAL_CFG["box_text_colour"],
                        VISUAL_CFG["box_text_scale"],
                        VISUAL_CFG["box_text_thickness"],
                        VISUAL_CFG["box_text_bg"],
                        0.8,  # slightly stronger bg for box text
                    )

                keypoints = []
                if i < len(kpts_xyn):
                    # --- extract pixel keypoints and validity (handles vis==0.0 or [0,0]) ---
                    k_pix, k_valid = _extract_kpts_pix_and_valid(kpts_xyn[i], img_w, img_h)

                    # keypoints (always) -> but draw only valid ones
                    for j, (pt, ok) in enumerate(zip(k_pix, k_valid)):
                        if not ok:
                            continue
                        kx, ky = int(pt[0]), int(pt[1])
                        cv2.circle(img_vis, (kx, ky), VISUAL_CFG["kpt_radius"], VISUAL_CFG["kpt_colour"], VISUAL_CFG["kpt_thickness"])

                    # preserve label file structure (unchanged)
                    for (kx_n, ky_n) in kpts_xyn[i][..., :2]:
                        keypoints.extend([kx_n, ky_n, 2])

                    # ---------- PBL (mesial/distal) ----------
                    if k_pix.shape[0] >= 7:
                        rot = (rotate_pred[i] if (isinstance(rotate_pred, (list, tuple, np.ndarray)) and i < len(rotate_pred))
                               else 0.0)
                        rot_rad = math.radians(rot)

                        # --- NEW: side separation sanity-check (10% avg tooth width) ---
                        side_sep_ok = True
                        if avg_box_w is not None and avg_box_w > 0:
                            cej_m_pt = k_pix[0] if (k_valid.size > 0 and k_valid[0]) else None
                            bl_m_pt  = k_pix[1] if (k_valid.size > 1 and k_valid[1]) else None
                            cej_d_pt = k_pix[3] if (k_valid.size > 3 and k_valid[3]) else None
                            bl_d_pt  = k_pix[4] if (k_valid.size > 4 and k_valid[4]) else None
                            if (cej_m_pt is not None and cej_d_pt is not None and
                                bl_m_pt  is not None and bl_d_pt  is not None):
                                cej_dist = float(np.linalg.norm(np.array(cej_m_pt) - np.array(cej_d_pt)))
                                bl_dist  = float(np.linalg.norm(np.array(bl_m_pt)  - np.array(bl_d_pt)))
                                thresh = 0.10 * float(avg_box_w)
                                side_sep_ok = (cej_dist >= thresh) and (bl_dist >= thresh)

                        # MESIAL
                        if k_valid.size > 6 and k_valid[0] and k_valid[1] and side_sep_ok:
                            cej_m, bl_m = k_pix[0], k_pix[1]
                            rl_m1 = k_pix[5] if k_valid[5] else np.array([np.nan, np.nan])
                            rl_m2 = k_pix[6] if k_valid[6] else np.array([np.nan, np.nan])
                            rl_m3 = np.array([np.nan, np.nan], dtype=float)

                            pbl_m_vals, _ = calculate_bone_loss(cej_m, bl_m, rl_m1, rl_m2, rl_m3, rot, img_vis.copy())
                            m_vals = np.array(pbl_m_vals, dtype=float)
                            valid_m = np.where(m_vals >= 0.0)[0]
                            m_best, m_idx_global = None, None
                            if valid_m.size > 0:
                                m_idx = valid_m[np.argmax(m_vals[valid_m])]
                                m_idx_global = [0,1,2][m_idx]
                                m_best = m_vals[m_idx]
                            m_sev  = _pbl_severity_class(m_best)
                            col_m  = VISUAL_CFG["pbl_colour_map"].get(m_sev, VISUAL_CFG["pbl_line_colour"])
                            # --- NEW: severity on BACKGROUND; text colour optionally overridden per severity ---
                            txt_bg_m = VISUAL_CFG.get("pbl_text_bg_map", VISUAL_CFG["pbl_colour_map"]).get(m_sev, VISUAL_CFG["pbl_text_bg"])
                            txt_col_m = VISUAL_CFG.get("pbl_text_fg_map", {}).get(m_sev, VISUAL_CFG["pbl_text_colour"])

                            bl_m_int = find_intersection(cej_m, bl_m, rot_rad)

                            rl_m_candidates = [rl_m1, rl_m2, rl_m3]
                            if m_idx_global is not None:
                                rl_m_sel_raw = rl_m_candidates[m_idx_global]
                                if not (np.isnan(rl_m_sel_raw[0]) or np.isnan(rl_m_sel_raw[1])):
                                    rl_m_int = find_intersection(cej_m, rl_m_sel_raw, rot_rad)

                                    # only draw BL dashed when also drawing other PBL lines
                                    _draw_dashed_line(
                                        img_vis,
                                        (int(bl_m[0]), int(bl_m[1])),
                                        (int(bl_m_int[0]), int(bl_m_int[1])),
                                        VISUAL_CFG["pbl_dashed_colour"],
                                        VISUAL_CFG["pbl_dashed_thickness"]
                                    )

                                    cv2.line(img_vis, (int(cej_m[0]), int(cej_m[1])), (int(bl_m_int[0]), int(bl_m_int[1])), col_m, VISUAL_CFG["pbl_line_thickness"])
                                    cv2.line(img_vis, (int(cej_m[0]), int(cej_m[1])), (int(rl_m_int[0]), int(rl_m_int[1])), col_m, VISUAL_CFG["pbl_line_thickness"])
                                    _draw_dashed_line(img_vis, (int(rl_m_sel_raw[0]), int(rl_m_sel_raw[1])), (int(rl_m_int[0]), int(rl_m_int[1])), VISUAL_CFG["pbl_dashed_colour"], VISUAL_CFG["pbl_dashed_thickness"])
                                    cv2.circle(img_vis, (int(bl_m_int[0]), int(bl_m_int[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)
                                    cv2.circle(img_vis, (int(rl_m_int[0]), int(rl_m_int[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)

                            cej_m_lbl = (int(cej_m[0])+8, int(cej_m[1])-8)
                            # lbl_m = f"PBL(M): {m_sev}" + (f" ({m_best:.2f})" if m_best is not None and m_best>=0 else "")
                            lbl_m = (f"{m_sev} ({int(m_best*100)}%)" if m_best is not None and m_best>=0 else "")
                            cej_m_lbl = _find_non_overlapping_label_pos(
                                cej_m_lbl, lbl_m, img_vis.shape,
                                VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                occupied_label_rects, margin=2
                            )
                            _put_text_with_bg(
                                img_vis, lbl_m, cej_m_lbl,
                                txt_col_m, VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                txt_bg_m, VISUAL_CFG["pbl_text_bg_opacity"]
                            )

                        # DISTAL
                        if k_valid.size > 6 and k_valid[3] and k_valid[4] and side_sep_ok:
                            cej_d, bl_d = k_pix[3], k_pix[4]
                            rl_d1 = k_pix[5] if k_valid[5] else np.array([np.nan, np.nan])
                            rl_d2 = k_pix[6] if k_valid[6] else np.array([np.nan, np.nan])
                            rl_d3 = np.array([np.nan, np.nan], dtype=float)

                            pbl_d_vals, _ = calculate_bone_loss(cej_d, bl_d, rl_d1, rl_d2, rl_d3, rot, img_vis.copy())
                            d_vals = np.array(pbl_d_vals, dtype=float)
                            valid_d = np.where(d_vals >= 0.0)[0]
                            d_best, d_idx_global = None, None
                            if valid_d.size > 0:
                                d_idx = valid_d[np.argmax(d_vals[valid_d])]
                                d_idx_global = [0,1,2][d_idx]
                                d_best = d_vals[d_idx]
                            d_sev  = _pbl_severity_class(d_best)
                            col_d  = VISUAL_CFG["pbl_colour_map"].get(d_sev, VISUAL_CFG["pbl_line_colour"])
                            # --- NEW: severity on BACKGROUND; text colour optionally overridden per severity ---
                            txt_bg_d = VISUAL_CFG.get("pbl_text_bg_map", VISUAL_CFG["pbl_colour_map"]).get(d_sev, VISUAL_CFG["pbl_text_bg"])
                            txt_col_d = VISUAL_CFG.get("pbl_text_fg_map", {}).get(d_sev, VISUAL_CFG["pbl_text_colour"])

                            bl_d_int = find_intersection(cej_d, bl_d, rot_rad)

                            rl_d_candidates = [rl_d1, rl_d2, rl_d3]
                            if d_idx_global is not None:
                                rl_d_sel_raw = rl_d_candidates[d_idx_global]
                                if not (np.isnan(rl_d_sel_raw[0]) or np.isnan(rl_d_sel_raw[1])):
                                    rl_d_int = find_intersection(cej_d, rl_d_sel_raw, rot_rad)

                                    # only draw BL dashed when also drawing other PBL lines
                                    _draw_dashed_line(
                                        img_vis,
                                        (int(bl_d[0]), int(bl_d[1])),
                                        (int(bl_d_int[0]), int(bl_d_int[1])),
                                        VISUAL_CFG["pbl_dashed_colour"],
                                        VISUAL_CFG["pbl_dashed_thickness"]
                                    )

                                    cv2.line(img_vis, (int(cej_d[0]), int(cej_d[1])), (int(bl_d_int[0]), int(bl_d_int[1])), col_d, VISUAL_CFG["pbl_line_thickness"])
                                    cv2.line(img_vis, (int(cej_d[0]), int(cej_d[1])), (int(rl_d_int[0]), int(rl_d_int[1])), col_d, VISUAL_CFG["pbl_line_thickness"])
                                    _draw_dashed_line(img_vis, (int(rl_d_sel_raw[0]), int(rl_d_sel_raw[1])), (int(rl_d_int[0]), int(rl_d_int[1])), VISUAL_CFG["pbl_dashed_colour"], VISUAL_CFG["pbl_dashed_thickness"])
                                    cv2.circle(img_vis, (int(bl_d_int[0]), int(bl_d_int[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)
                                    cv2.circle(img_vis, (int(rl_d_int[0]), int(rl_d_int[1])), VISUAL_CFG["pbl_marker_radius"], VISUAL_CFG["pbl_marker_colour"], -1)

                            cej_d_lbl = (int(cej_d[0])+8, int(cej_d[1])-8)
                            # lbl_d = f"PBL(D): {d_sev}" + (f" ({d_best:.2f})" if d_best is not None and d_best>=0 else "")
                            lbl_d = (f"{d_sev} ({int(d_best*100)}%)" if d_best is not None and d_best>=0 else "")
                            cej_d_lbl = _find_non_overlapping_label_pos(
                                cej_d_lbl, lbl_d, img_vis.shape,
                                VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                occupied_label_rects, margin=2
                            )
                            _put_text_with_bg(
                                img_vis, lbl_d, cej_d_lbl,
                                txt_col_d, VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                txt_bg_d, VISUAL_CFG["pbl_text_bg_opacity"]
                            )

                        # ---------- Furcation involvement (FA, FBL-m, FBL-d) ----------
                        # We assume indices: FA=7, FBL-m=8, FBL-d=9 if present.
                        def _ok(idx):
                            return (idx < k_valid.size) and k_valid[idx]

                        fa_pt   = k_pix[7] if (k_pix.shape[0] > 7 and _ok(7)) else None
                        fbl_m_pt = k_pix[8] if (k_pix.shape[0] > 8 and _ok(8)) else None
                        fbl_d_pt = k_pix[9] if (k_pix.shape[0] > 9 and _ok(9)) else None

                        if fa_pt is not None:
                            # choose FA line colour by classification
                            fa_label, _ = _classify_furcation(fa_pt, fbl_m_pt, fbl_d_pt, avg_box_w, ratio_thresh=0.05)
                            if fa_label == "Healthy":
                                fa_col = VISUAL_CFG["fa_colour_healthy"]
                            elif fa_label == "Involved":
                                fa_col = VISUAL_CFG["fa_colour_involved"]
                            else:
                                fa_col = VISUAL_CFG["pbl_line_colour"]  # fallback

                            # join FA to FBL-m / FBL-d if present
                            if fbl_m_pt is not None:
                                cv2.line(img_vis, (int(fa_pt[0]), int(fa_pt[1])), (int(fbl_m_pt[0]), int(fbl_m_pt[1])), fa_col, VISUAL_CFG["pbl_line_thickness"])
                            if fbl_d_pt is not None:
                                cv2.line(img_vis, (int(fa_pt[0]), int(fa_pt[1])), (int(fbl_d_pt[0]), int(fbl_d_pt[1])), fa_col, VISUAL_CFG["pbl_line_thickness"])

                            # FA label text (use FA-specific map)
                            if fa_label is not None:
                                fa_org = (int(fa_pt[0]) + 8, int(fa_pt[1]) - 8)
                                fa_org = _find_non_overlapping_label_pos(
                                    fa_org, fa_label, img_vis.shape,
                                    VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                    occupied_label_rects, margin=2
                                )
                                # use new background map for FA, with optional per-class text fg override
                                fa_text_bg = VISUAL_CFG.get("fa_text_bg_map", {}).get(fa_label, VISUAL_CFG["pbl_text_bg"])
                                fa_text_op = VISUAL_CFG.get("fa_text_bg_opacity", VISUAL_CFG["pbl_text_bg_opacity"])
                                fa_text_col = VISUAL_CFG.get("fa_text_fg_map", {}).get(
                                    fa_label, VISUAL_CFG["fa_text_colour_map"].get(fa_label, VISUAL_CFG["pbl_text_colour"])
                                )
                                _put_text_with_bg(
                                    img_vis, fa_label, fa_org,
                                    fa_text_col, VISUAL_CFG["pbl_text_scale"], VISUAL_CFG["pbl_text_thickness"],
                                    fa_text_bg, fa_text_op
                                )

                # write YOLO line (unchanged)
                f.write(" ".join(map(str, [cls_idx, cx_n, cy_n, w_n, h_n] + keypoints)) + "\n")

        save_vis = os.path.join(vis_root, img.rsplit(".", 1)[0] + "_predictions.jpg")
        cv2.imwrite(save_vis, img_vis)
        saved_images.append(save_vis)

    return {"saved_images": saved_images}