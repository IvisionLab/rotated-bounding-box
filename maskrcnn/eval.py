import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from maskrcnn.utils import compute_overlaps_masks




# create mask from rodated bounding-box
def rotated_box_mask(shape, rotated_boxes):
  mask = np.zeros(shape, dtype=np.uint8)
  for rbox in rotated_boxes:
    rbox = np.reshape(rbox, (-1, 2))
    rbox = rbox.astype(int)
    cv2.fillConvexPoly(mask, rbox, 255)
  return mask


# calculate the IoU between two mask
def calc_iou_from_masks(gt, dt):
  inter = np.zeros(gt.shape, dtype=np.uint8)
  cv2.bitwise_and(gt, dt, inter)
  inter_area = np.sum(inter) / 255.
  gt_area = np.sum(gt) / 255.
  dt_area = np.sum(dt) / 255.
  return inter_area / (gt_area + dt_area - inter_area)


# calculate IoU between annotations and detections
def calc_ious(annotations, detections, image_shape, nb_class):
  ious = []
  for cls_id in range(nb_class):
    dts = [dt for dt in detections if dt['id'] == cls_id]
    gts = [gt for gt in annotations if gt['id'] == cls_id]

    if not dts and not gts:
      continue

    dt_mask = rotated_box_mask(
        image_shape,
        [dt['rbox'] for dt in dts if not dt is None and len(dt) > 0])

    gt_mask = rotated_box_mask(
        image_shape,
        [gt['rbox'] for gt in gts if not gt is None and len(gt) > 0])

    iou = calc_iou_from_masks(gt_mask, dt_mask)
    ious = [iou]
  return ious


# calculate average recall based on IoU
def average_recall(iou):
  all_iou = sorted(iou)
  num_pos = len(all_iou)
  dx = 0.001

  overlap = np.arange(0, 1.0, dx)
  overlap[-1] = 1

  N = len(overlap)
  recall = np.zeros(N, dtype=np.float32)
  for i in range(N):
      recall[i] = (all_iou > overlap[i]).sum() / float(num_pos) if not num_pos is None and num_pos > 0 else 0

  good_recall = recall[np.where(overlap > 0.5)]
  AR = 2 * dx * np.trapz(good_recall)
  return overlap, recall, AR


def plot_average_recall(ious):
  overlap, recall, AR = average_recall(ious)
  print("Average Recall: {}".format(AR))
  plt.figure()
  plt.step(overlap, recall, color='b', alpha=0.2, where='post')
  plt.fill_between(overlap, recall, step='post', alpha=0.2, color='b')
  plt.xlabel('IoU')
  plt.ylabel('Recall')
  plt.ylim([0.0, 1.05])
  plt.xlim([0.0, 1.0])


def masks_from_rotated_boxes(rboxes, image_shape):
  masks = []
  if len(rboxes) > 0:
    for rbox in rboxes:
      mask = rotated_box_mask(image_shape, [rbox])
      masks += [mask]
  else:
    mask = np.zeros(image_shape, dtype=np.uint8)
    masks += [mask]

  masks = np.stack(masks, axis=2)
  return masks

def bbox_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def compute_overlaps_bboxes(gt_boxes, pred_boxes):
  overlaps = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]))
  for i in range(pred_boxes.shape[0]):
    for j in range(gt_boxes.shape[0]):
      overlaps[i, j] = bbox_iou(pred_boxes[i], gt_boxes[j])

  return overlaps


def compute_match(annotations, detections, image_shape, evaltype="segm", iou_threshold=0.5):
  gt_class_ids = np.array([gt['id'] for gt in annotations])
  gt_bboxes = np.array([gt['bbox'] for gt in annotations])
  gt_rboxes = np.array([gt['rbox'] for gt in annotations])
  pred_class_ids = np.array([dt['id'] for dt in detections])
  pred_scores = np.array([dt['score'] for dt in detections])
  pred_bboxes = np.array([dt['bbox'] for dt in detections])
  pred_rboxes = np.array([dt['rbox'] for dt in detections])

  indices = np.argsort(pred_scores)[::-1]
  pred_class_ids = pred_class_ids[indices]
  pred_scores = pred_scores[indices]
  pred_rboxes = pred_rboxes[indices]

  if evaltype == "segm":
    gt_masks = masks_from_rotated_boxes(gt_rboxes, image_shape)
    pred_masks = masks_from_rotated_boxes(pred_rboxes, image_shape)
    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_overlaps_masks(pred_masks, gt_masks)
  elif evaltype == "bbox":
    overlaps = compute_overlaps_bboxes(gt_bboxes, pred_bboxes)

  # Loop through ground truth boxes and find matching predictions
  pred_match = np.zeros([pred_rboxes.shape[0]])
  gt_match = np.zeros([gt_rboxes.shape[0]])
  for i in range(len(pred_rboxes)):
    # Find best matching ground truth box
    sorted_ixs = np.argsort(overlaps[i])[::-1]
    for j in sorted_ixs:
      # If ground truth box is already matched, go to next one
      if gt_match[j] == 1:
        continue
      # If we reach IoU smaller than the threshold, end the loop
      iou = overlaps[i, j]
      if iou < iou_threshold:
        break
      # Do we have a match?
      if pred_class_ids[i] == gt_class_ids[j]:
        gt_match[j] = 1
        pred_match[i] = 1
        break

  return pred_match, gt_match, pred_scores, overlaps


# load matches
def load_matches(filepath):
  targets = []
  scores = []
  overlaps = []
  ious = []

  with open(filepath, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    for row in csv_reader:
      targets += [int(row[0])]
      scores += [float(row[1])]
      overlaps += [float(row[2])]
      ious += [float(row[3])]

  targets = np.array(targets)
  scores = np.array(scores)
  overlaps = np.array(overlaps)
  ious = np.array(ious)

  return targets, scores, overlaps, ious


def compute_ap(dt_matches):
  precisions = np.cumsum(dt_matches).astype(np.float32) / (np.arange(len(dt_matches)) + 1)
  recalls = np.cumsum(dt_matches).astype(np.float32) / np.sum(dt_matches)

  # Pad with start and end values to simplify the math
  precisions = np.concatenate([[0], precisions, [0]])
  recalls = np.concatenate([[0], recalls, [1]])

  for i in range(len(precisions) - 2, -1, -1):
    precisions[i] = np.maximum(precisions[i], precisions[i + 1])

  # Compute mean AP over recall range
  indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
  mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
  return precisions, recalls, mAP