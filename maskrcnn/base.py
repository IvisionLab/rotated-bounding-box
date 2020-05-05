import os
import sys
import json
import datetime
import re
import math
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import maskrcnn
from maskrcnn import model, utils
from maskrcnn.model import rpn_class_loss_graph, rpn_bbox_loss_graph
from maskrcnn.model import log, fullmatch


def unmold_detections(dts, image_shape, window, config):
  zero_ix = np.where(dts[:, 0] == 0)[0]
  N = zero_ix[0] if zero_ix.shape[0] > 0 else dts.shape[0]
  class_ids = dts[:N, 0].astype(np.int32)
  scores = dts[:N, 1]
  boxes = dts[:N, 2:6]

  # Compute scale and shift to translate coordinates to image domain.
  h_scale = image_shape[0] / (window[2] - window[0])
  w_scale = image_shape[1] / (window[3] - window[1])

  scale = min(h_scale, w_scale)
  shift = window[:2]  # y, x
  scales = np.array([scale, scale, scale, scale])
  shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

  # Filter out detections with zero area. Often only happens in early
  # stages of training when the network weights are still a bit random.
  exclude_ix = np.where(
      (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]

  # Translate bounding boxes to image domain
  boxes = np.multiply(boxes - shifts, scales).astype(np.int32)
  if config.regressor == "deltas":
    tlines = dts[:N, 6:10]
    tlines = np.multiply(tlines - shifts, scales).astype(np.int32)
    rboxes_verts = top_line_to_vertices(tlines, boxes)
  elif config.regressor == "rotdim":
    angles = dts[:N, 6]
    dimensions = dts[:N, 7:9]
    dimensions = np.multiply(dimensions, [scale, scale]).astype(np.int32)
    angles *= 180
    angles -= 90
    rboxes_verts = rotated_box_to_vertices(angles, dimensions, boxes)
  elif config.regressor == "verts":
    verts = dts[:N,6:14]
    scales = np.repeat(scales, 2)
    shifts = np.hstack((shifts, shifts))
    verts = np.multiply(verts-shifts, scales).astype(np.int32)
    rboxes_verts = refine_vertices(verts, boxes)


  outputs = []
  if exclude_ix.shape[0] > 0:
    boxes = np.delete(boxes, exclude_ix, axis=0)
    class_ids = np.delete(class_ids, exclude_ix, axis=0)
    scores = np.delete(scores, exclude_ix, axis=0)

    if config.regressor == "deltas" or \
        config.regressor == "rotdim" or \
        config.regressor == "verts":
      rboxes_verts = np.delete(rboxes_verts, exclude_ix, axis=0)

  outputs = [class_ids, scores, boxes]

  if config.regressor == "deltas" or \
      config.regressor == "rotdim" or \
      config.regressor == "verts":
    outputs += [rboxes_verts]

  return outputs

def refine_vertices(verts, boxes):
  cy = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) * 0.5
  cx = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * 0.5

  w = np.zeros([verts.shape[0]])
  h = np.zeros([verts.shape[0]])
  angles = np.zeros([verts.shape[0]])

  for i in range(verts.shape[0]):
    y1, x1, y2, x2, y3, x3 = verts[i,0:6]
    h[i] = math.sqrt(math.pow(x3 - x2, 2) + math.pow(y3 - y2, 2))
    w[i] = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
    dx = x2-x1
    dy = y2-y1
    rad = math.atan2(dy, dx)
    angles[i] = math.degrees(rad)

  return utils.rboxes2points(cy, cx, h, w, angles)


def rotated_box_to_vertices(angles, dimensions, boxes):
  h = dimensions[:, 0]
  w = dimensions[:, 1]
  cy = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) * 0.5
  cx = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) * 0.5
  rboxes_verts = utils.rboxes2points(cy, cx, h, w, angles)
  return rboxes_verts


def top_line_to_vertices(tlines, boxes):
  y1 = tlines[:, 0]
  x1 = tlines[:, 1]
  y2 = tlines[:, 2]
  x2 = tlines[:, 3]
  y3 = boxes[:, 2] - (y1 - boxes[:, 0])
  x3 = boxes[:, 3] - (x1 - boxes[:, 1])
  y4 = boxes[:, 0] + (boxes[:, 2] - y2)
  x4 = boxes[:, 1] + (boxes[:, 3] - x2)
  return np.stack([y1, x1, y2, x2, y3, x3, y4, x4], axis=1)


def vertices_fliplr(verts):
  flipped = []
  for v in verts:
    v = v.reshape(-1, 2)
    v = np.fliplr(v)
    flipped += [v.reshape(-1)]
  return flipped


def line_to_deltas(box, rbox):
  box = tf.cast(box, tf.float32)
  rbox = tf.cast(rbox, tf.float32)

  height = box[:, 2] - box[:, 0]
  width = box[:, 3] - box[:, 1]

  dy1 = (rbox[:, 0] - box[:, 0]) / height
  dx1 = (rbox[:, 1] - box[:, 1]) / width
  dy2 = (box[:, 2] - rbox[:, 2]) / height
  dx2 = (box[:, 3] - rbox[:, 3]) / width

  result = tf.stack([dy1, dx1, dy2, dx2], axis=1)
  return result


def deltas_to_line_graph(boxes, deltas):
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  y1 = deltas[:, 0] * height + boxes[:, 0]
  x1 = deltas[:, 1] * width + boxes[:, 1]
  y2 = boxes[:, 2] - deltas[:, 2] * height
  x2 = boxes[:, 3] - deltas[:, 3] * width
  result = tf.stack([y1, x1, y2, x2], axis=1, name="deltas_to_line_out")
  return result


def clip_line_graph(line, boxes):
  wy1 = boxes[:, 0]
  wx1 = boxes[:, 1]
  wy2 = boxes[:, 2]
  wx2 = boxes[:, 3]
  y1 = tf.maximum(tf.minimum(line[:, 0], wy2), wy1)
  x1 = tf.maximum(tf.minimum(line[:, 1], wx2), wx1)
  y2 = tf.maximum(tf.minimum(line[:, 2], wy2), wy1)
  x2 = tf.maximum(tf.minimum(line[:, 3], wx2), wx1)
  return tf.stack([y1, x1, y2, x2], axis=1, name="clipped_line")


def check_image_size(config):
  # Image size must be dividable by 2 multiple times
  h, w = config.IMAGE_SHAPE[:2]
  if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
    raise Exception("Image size must be dividable by 2 at least 6 times "
                    "to avoid fractions when downscaling and upscaling."
                    "For example, use 256, 320, 384, 448, 512, ... etc. ")


def upsample_layer(M, M_name, C, C_name):
  return [
      KL.UpSampling2D(size=(2, 2), name=M_name)(M),
      KL.Conv2D(256, (1, 1), name=C_name)(C)
  ]


def feature_extractor_layers(input_image, config):
  # Bottom-up Layers
  _, C2, C3, C4, C5 = rboxnet.model.resnet_graph(
      input_image, config.BACKBONE, stage5=True)

  # Top-down Layers
  M5 = KL.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
  M4 = KL.Add(name="fpn_p4add")(upsample_layer(M5, "fpn_p5upsampled", C4,
                                               "fpn_c4p4"))
  M3 = KL.Add(name="fpn_p3add")(upsample_layer(M4, "fpn_p4upsampled", C3,
                                               "fpn_c3p3"))
  M2 = KL.Add(name="fpn_p2add")(upsample_layer(M3, "fpn_p3upsampled", C2,
                                               "fpn_c2p2"))

  # Attach 3x3 conv to all P layers to get the final feature maps.
  P2 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p2")(M2)
  P3 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p3")(M3)
  P4 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p4")(M4)
  P5 = KL.Conv2D(256, (3, 3), padding="SAME", name="fpn_p5")(M5)
  # P6 is used for the 5th anchor scale in RPN.
  # Generated by subsampling from P5 with stride of 2.
  P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)

  return P2, P3, P4, P5, P6


def rpn_layers(rpn_feature_maps, anchors, proposal_count, config):
  # RPN Model
  rpn = model.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                              len(config.RPN_ANCHOR_RATIOS), 256)

  # Loop through pyramid layers
  output_layers = []
  for p in rpn_feature_maps:
    output_layers.append(rpn([p]))

  # Concatenate layer outputs
  # Convert from list of lists of level outputs to list of lists of outputs across levels.
  # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
  output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
  outputs = list(zip(*output_layers))
  rpn_class_logits, rpn_class, rpn_bbox = [
      KL.Concatenate(axis=1, name=n)(list(o))
      for o, n in zip(outputs, output_names)
  ]

  # Generate proposals
  # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates and zero padded
  rpn_rois = model.ProposalLayer(
      proposal_count=proposal_count,
      nms_threshold=config.RPN_NMS_THRESHOLD,
      name="ROI",
      anchors=anchors,
      config=config)([rpn_class, rpn_bbox])

  return [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois]


def rbox_class_loss_graph(target_class_ids, pred_class_logits,
                          active_class_ids):
  target_class_ids = tf.cast(target_class_ids, 'int64')

  # Find predictions of classes that are not in the dataset.
  pred_class_ids = tf.argmax(pred_class_logits, axis=2)
  pred_active = tf.gather(active_class_ids[0], pred_class_ids)

  # Loss
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=target_class_ids, logits=pred_class_logits)

  # Erase losses of predictions of classes that are not in the active
  # classes of the image.
  loss = loss * pred_active

  # Computer loss mean. Use only predictions that contribute
  # to the loss to get a correct mean.
  loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
  return loss


def rbox_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
  # Reshape to merge batch and roi dimensions for simplicity.
  target_class_ids = K.reshape(target_class_ids, (-1, ))
  target_bbox = K.reshape(target_bbox, (-1, 4))
  pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 4))

  # Only positive ROIs contribute to the loss. And only
  # the right class_id of each ROI. Get their indicies.
  positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
  positive_roi_class_ids = tf.cast(
      tf.gather(target_class_ids, positive_roi_ix), tf.int64)
  indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

  # Gather the deltas (predicted and true) that contribute to loss
  target_bbox = tf.gather(target_bbox, positive_roi_ix)
  pred_bbox = tf.gather_nd(pred_bbox, indices)

  # Smooth-L1 Loss
  loss = K.switch(
      tf.size(target_bbox) > 0,
      model.smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox),
      tf.constant(0.0))
  loss = K.mean(loss)
  loss = K.reshape(loss, [1, 1])
  return loss


def rpn_class_loss_layer(input_rpn_match, rpn_class_logits):
  return KL.Lambda(
      lambda x: rpn_class_loss_graph(*x),
      name="rpn_class_loss")([input_rpn_match, rpn_class_logits])


def rpn_bbox_loss_layer(config, input_rpn_bbox, input_rpn_match, rpn_bbox):
  return KL.Lambda(
      lambda x: rpn_bbox_loss_graph(config, *x),
      name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])


def rbox_class_loss_layer(target_class_ids, rbox_class_logits,
                          active_class_ids):
  return KL.Lambda(
      lambda x: rbox_class_loss_graph(*x), name="rbox_class_loss")(
          [target_class_ids, rbox_class_logits, active_class_ids])


def rbox_bbox_loss_layer(target_bbox, target_class_ids, rbox_bbox):
  return KL.Lambda(
      lambda x: rbox_bbox_loss_graph(*x),
      name="rbox_bbox_loss")([target_bbox, target_class_ids, rbox_bbox])


def rbox_deltas_loss_layer(target_rbox_deltas, target_class_ids, rbox_deltas):
  return KL.Lambda(
      lambda x: model.custom_loss_graph(*x), name="rbox_deltas_loss")(
          [target_rbox_deltas, target_class_ids, rbox_deltas])


def rbox_angles_loss_layer(target_rbox_angles, target_class_ids, rbox_angles):
  return KL.Lambda(
      lambda x: model.custom_loss_graph(*x), name="rbox_angles_loss")(
          [target_rbox_angles, target_class_ids, rbox_angles])


def rbox_dim_loss_layer(target_rbox_dim, target_class_ids, rbox_dim):
  return KL.Lambda(
      lambda x: model.custom_loss_graph(*x),
      name="rbox_dim_loss")([target_rbox_dim, target_class_ids, rbox_dim])


def rbox_verts_loss_layer(target_rbox_verts, target_class_ids, rbox_verts):
  return KL.Lambda(
      lambda x: model.custom_loss_graph(*x), name="rbox_verts_loss")(
          [target_rbox_verts, target_class_ids, rbox_verts])


def roi_align_layers(rois, feature_maps, config):
  return model.PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                               config.IMAGE_SHAPE,
                               name="roi_align")([rois] + feature_maps)


def full_connected_layers(x, config):
  # Two 1024 FC layers (implemented with Conv2D for consistency)
  x = KL.TimeDistributed(
      KL.Conv2D(1024, (config.POOL_SIZE, config.POOL_SIZE), padding="valid"),
      name="rbox_class_conv1")(x)

  x = KL.TimeDistributed(model.BatchNorm(axis=3), name='rbox_class_bn1')(x)
  x = KL.Activation('relu')(x)
  x = KL.TimeDistributed(KL.Conv2D(1024, (1, 1)), name="rbox_class_conv2")(x)
  x = KL.TimeDistributed(model.BatchNorm(axis=3), name='rbox_class_bn2')(x)
  x = KL.Activation('relu')(x)
  return KL.Lambda(
      lambda x: K.squeeze(K.squeeze(x, 3), 2), name="pool_squeeze")(x)


############################################################
#  Feature Pyramid Network Heads
############################################################
def bbox_classifier_layers(shared, num_classes):
  # Classifier head
  rbox_class_logits = KL.TimeDistributed(
      KL.Dense(num_classes), name='rbox_class_logits')(shared)
  rbox_probs = KL.TimeDistributed(
      KL.Activation("softmax"), name="rbox_class")(rbox_class_logits)

  # BBox head
  # [batch, boxes, num_classes * (dy, dx, log(dh), log(dw))]
  x = KL.TimeDistributed(
      KL.Dense(num_classes * 4, activation='linear'),
      name='rbox_bbox_fc')(shared)

  # Reshape to [batch, boxes, num_classes, (dy, dx, log(dh), log(dw))]
  s = K.int_shape(x)
  rbox_bbox = KL.Reshape((s[1], num_classes, 4), name="rbox_bbox")(x)

  return rbox_class_logits, rbox_probs, rbox_bbox


def rbox_deltas_regressor_layers(shared, num_classes):
  # rotated bounding box topline deltas
  x = KL.TimeDistributed(
      KL.Dense(num_classes * 4, activation='linear'),
      name='rbox_deltas_fc')(shared)

  s = K.int_shape(x)
  rbox_deltas = KL.Reshape((s[1], num_classes, 4), name="rbox_deltas")(x)
  return rbox_deltas


def rbox_rotdim_regressor_layers(shared, num_classes):
  # angles regressor
  x = KL.TimeDistributed(
      KL.Dense(num_classes * 1, activation='linear'),
      name='rbox_angles_fc')(shared)

  s = K.int_shape(x)
  rbox_angles = KL.Reshape((s[1], num_classes, 1), name="rbox_angles")(x)

  # dimension (width and height) regressor
  x = KL.TimeDistributed(
      KL.Dense(num_classes * 2, activation='linear'),
      name='rbox_dim_fc')(shared)

  s = K.int_shape(x)
  rbox_dim = KL.Reshape((s[1], num_classes, 2), name="rbox_dim")(x)

  return rbox_angles, rbox_dim


def rbox_verts_regressor_layers(shared, num_classes):
  # rotated bounding vertices
  x = KL.TimeDistributed(
      KL.Dense(num_classes * 8, activation='linear'),
      name='rbox_verts_fc')(shared)

  s = K.int_shape(x)
  rbox_verts = KL.Reshape((s[1], num_classes, 8), name="rbox_verts")(x)
  return rbox_verts


def detection_target_layer(config, inputs):
  return DetectionTargetLayer(config, name="proposal_targets")(inputs)


def detection_layer(config, inputs):
  return DetectionLayer(config, name="rbox_detection")(inputs)


def detection_targets_graph(config, proposals, gt_class_ids, gt_boxes,
                            **kwargs):
  # Assertions
  asserts = \
      [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals], name="roi_assertion"),]

  with tf.control_dependencies(asserts):
    proposals = tf.identity(proposals)

  # Remove zero padding
  proposals, _ = \
      model.trim_zeros_graph(proposals, name="trim_proposals")
  gt_boxes, non_zeros = \
      model.trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
  gt_class_ids = \
      model.tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")

  # Handle COCO crowds
  # A crowd box in COCO is a bounding box around several instances. Exclude
  # them from training. A crowd box is given a negative class ID.
  crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
  non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
  crowd_boxes = tf.gather(gt_boxes, crowd_ix)
  gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
  gt_boxes = tf.gather(gt_boxes, non_crowd_ix)

  # Compute overlaps matrix [proposals, gt_boxes]
  overlaps = model.overlaps_graph(proposals, gt_boxes)

  # Compute overlaps with crowd boxes [anchors, crowds]
  crowd_overlaps = model.overlaps_graph(proposals, crowd_boxes)
  crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
  no_crowd_bool = (crowd_iou_max < 0.001)

  # Determine postive and negative ROIs
  roi_iou_max = tf.reduce_max(overlaps, axis=1)
  # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
  positive_roi_bool = (roi_iou_max >= 0.5)
  positive_indices = tf.where(positive_roi_bool)[:, 0]
  # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
  negative_indices = tf.where(
      tf.logical_and(roi_iou_max < 0.5, no_crowd_bool))[:, 0]

  # Subsample ROIs. Aim for 33% positive
  # Positive ROIs
  positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
  positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
  positive_count = tf.shape(positive_indices)[0]
  # Negative ROIs. Add enough to maintain positive:negative ratio.
  r = 1.0 / config.ROI_POSITIVE_RATIO
  negative_count = tf.cast(r * tf.cast(positive_count, tf.float32),
                           tf.int32) - positive_count
  negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
  # Gather selected ROIs
  positive_rois = tf.gather(proposals, positive_indices)
  negative_rois = tf.gather(proposals, negative_indices)

  # Assign positive ROIs to GT boxes.
  positive_overlaps = tf.gather(overlaps, positive_indices)
  roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
  roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
  roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

  # Compute bbox refinement for positive ROIs
  deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
  deltas /= config.BBOX_STD_DEV

  # Compute mask targets
  boxes = positive_rois

  # Append negative ROIs and pad bbox deltas and masks that
  # are not used for negative ROIs with zeros.
  rois = tf.concat([positive_rois, negative_rois], axis=0)
  N = tf.shape(negative_rois)[0]
  P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
  rois = tf.pad(rois, [(0, P), (0, 0)])
  roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
  deltas = tf.pad(deltas, [(0, N + P), (0, 0)])

  if 'gt_rboxes' in kwargs:
    gt_rboxes = kwargs.get('gt_rboxes')
    gt_rboxes, _ = model.trim_zeros_graph(gt_rboxes, name="trim_gt_rboxes")
    gt_rboxes = tf.gather(gt_rboxes, non_crowd_ix)
    roi_gt_rboxes = tf.gather(gt_rboxes, roi_gt_box_assignment)

    if config.regressor == "deltas":
      rbox_deltas = line_to_deltas(roi_gt_boxes, roi_gt_rboxes)
      rbox_deltas /= 0.2
      rbox_deltas = tf.pad(rbox_deltas, [(0, N + P), (0, 0)])
      return rois, roi_gt_class_ids, deltas, rbox_deltas
    elif config.regressor == "rotdim":
      gt_angles = kwargs.get('gt_angles')
      gt_angles = tf.boolean_mask(gt_angles, non_zeros, name="trim_gt_angles")
      gt_angles = tf.gather(gt_angles, non_crowd_ix)
      roi_gt_angles = tf.gather(gt_angles, roi_gt_box_assignment)
      rbox_angles = roi_gt_angles / 0.2

      # point 1
      y1 = roi_gt_rboxes[:, 0]
      x1 = roi_gt_rboxes[:, 1]
      # point 2
      y2 = roi_gt_rboxes[:, 2]
      x2 = roi_gt_rboxes[:, 3]
      # point 3
      y3 = roi_gt_rboxes[:, 4]
      x3 = roi_gt_rboxes[:, 5]

      rw = tf.sqrt(tf.pow(x2 - x1, 2) + tf.pow(y2 - y1, 2))
      rh = tf.sqrt(tf.pow(x3 - x2, 2) + tf.pow(y3 - y2, 2))
      rbox_dim = tf.stack([rh, rw], axis=1)
      rbox_dim /= 0.1

      rbox_dim = tf.pad(rbox_dim, [(0, N + P), (0, 0)])
      rbox_angles = tf.pad(rbox_angles, [(0, N + P), (0, 0)])
      return rois, roi_gt_class_ids, deltas, rbox_angles, rbox_dim
    elif config.regressor == "verts":
      roi_gt_rboxes = tf.pad(roi_gt_rboxes, [(0, N + P), (0, 0)])
      return rois, roi_gt_class_ids, deltas, roi_gt_rboxes

  return rois, roi_gt_class_ids, deltas


def refine_detections_graph(rois, probs, deltas, window, config, **kwargs):
  # Class IDs per ROI
  class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
  # Class probability of the top class of each ROI
  indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
  class_scores = tf.gather_nd(probs, indices)
  # Class-specific bounding box deltas
  deltas_specific = tf.gather_nd(deltas, indices)
  # Apply bounding box deltas
  # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
  norm_rois = model.apply_box_deltas_graph(
      rois, deltas_specific * config.BBOX_STD_DEV)
  # Convert coordiates to image domain
  height, width = config.IMAGE_SHAPE[:2]
  scaled_rois = norm_rois * tf.constant([height, width, height, width],
                                        dtype=tf.float32)
  # Clip boxes to image window
  clipped_rois = model.clip_boxes_graph(scaled_rois, window)
  # Round and cast to int since we're deadling with pixels now
  refined_rois = tf.to_int32(tf.rint(clipped_rois))

  if config.regressor == "deltas" and not kwargs.get('rbox_deltas') == None:
    rbox_deltas = kwargs.get('rbox_deltas')
    rbox_deltas = tf.gather_nd(rbox_deltas, indices)
    rbox_line = deltas_to_line_graph(norm_rois, rbox_deltas * 0.2)
    rbox_line *= tf.constant([height, width, height, width], dtype=tf.float32)
    rbox_line = clip_line_graph(rbox_line, clipped_rois)
    rbox_line = tf.to_int32(tf.rint(rbox_line))
  elif config.regressor == "rotdim" and not kwargs.get('rbox_angles') == None and \
       not kwargs.get('rbox_dim') == None:
    rbox_angles = kwargs.get('rbox_angles')
    rbox_angles = tf.gather_nd(rbox_angles, indices)
    rbox_angles *= 0.2

    rbox_dim = kwargs.get('rbox_dim')
    rbox_dim = tf.gather_nd(rbox_dim, indices)
    rbox_dim *= 0.1
    rbox_dim *= tf.constant([height, width], dtype=tf.float32)
    rbox_dim = tf.to_int32(tf.rint(rbox_dim))
  elif config.regressor == "verts" and not kwargs.get('rbox_verts') == None:
    rbox_verts = kwargs.get('rbox_verts')
    rbox_verts = tf.gather_nd(rbox_verts, indices)
    rbox_verts *= tf.constant(
        [height, width, height, width, height, width, height, width],
        dtype=tf.float32)
    rbox_verts = model.clip_rotated_boxes_graph(rbox_verts, clipped_rois)
    rbox_verts = tf.to_int32(tf.rint(rbox_verts))

  # Filter out background boxes
  keep = tf.where(class_ids > 0)[:, 0]
  # Filter out low confidence boxes
  if config.DETECTION_MIN_CONFIDENCE:
    conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
    keep = tf.sets.set_intersection(
        tf.expand_dims(keep, 0), tf.expand_dims(conf_keep, 0))
    keep = tf.sparse_tensor_to_dense(keep)[0]

  # Apply per-class NMS
  # 1. Prepare variables
  pre_nms_class_ids = tf.gather(class_ids, keep)
  pre_nms_scores = tf.gather(class_scores, keep)
  pre_nms_rois = tf.gather(refined_rois, keep)
  unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

  def nms_keep_map(class_id):
    """Apply Non-Maximum Suppression on ROIs of the given class."""
    # Indices of ROIs of the given class
    ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
    # Apply NMS
    class_keep = tf.image.non_max_suppression(
        tf.to_float(tf.gather(pre_nms_rois, ixs)),
        tf.gather(pre_nms_scores, ixs),
        max_output_size=config.DETECTION_MAX_INSTANCES,
        iou_threshold=config.DETECTION_NMS_THRESHOLD)
    # Map indicies
    class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
    # Pad with -1 so returned tensors have the same shape
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
    class_keep = tf.pad(
        class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
    # Set shape so map_fn() can infer result shape
    class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
    return class_keep

  # 2. Map over class IDs
  nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int64)
  # 3. Merge results into one list, and remove -1 padding
  nms_keep = tf.reshape(nms_keep, [-1])
  nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
  # 4. Compute intersection between keep and nms_keep
  keep = tf.sets.set_intersection(
      tf.expand_dims(keep, 0), tf.expand_dims(nms_keep, 0))
  keep = tf.sparse_tensor_to_dense(keep)[0]
  # Keep top detections
  roi_count = config.DETECTION_MAX_INSTANCES
  class_scores_keep = tf.gather(class_scores, keep)
  num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
  top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
  keep = tf.gather(keep, top_ids)

  # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
  # Coordinates are in image domain.

  outputs = [
      tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
      tf.gather(class_scores, keep)[..., tf.newaxis],
      tf.to_float(tf.gather(refined_rois, keep))
  ]

  if config.regressor == "deltas" and not rbox_line == None:
    outputs += [tf.to_float(tf.gather(rbox_line, keep))]
  elif config.regressor == "rotdim" and not rbox_angles == None and not rbox_dim == None:
    outputs += [
        tf.to_float(tf.gather(rbox_angles, keep)),
        tf.to_float(tf.gather(rbox_dim, keep))
    ]
  elif config.regressor == "verts" and not rbox_verts == None:
    outputs += [tf.to_float(tf.gather(rbox_verts, keep))]

  detections = tf.concat(outputs, axis=1)

  # Pad with zeros if detections < DETECTION_MAX_INSTANCES
  gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
  detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
  return detections


############################################################
#  DetectionTargetLayer Class
############################################################
class DetectionTargetLayer(KE.Layer):
  def __init__(self, config, **kwargs):
    super(DetectionTargetLayer, self).__init__(**kwargs)
    self.config = config

  def call(self, inputs):
    names = ["rois", "target_class_ids", "target_bbox"]
    if self.config.regressor == "deltas":
      names += ["target_rbox_deltas"]
      outputs = utils.batch_slice(
          inputs,
          lambda a, b, c, d: detection_targets_graph(self.config, a, b, c, gt_rboxes=d),
          self.config.IMAGES_PER_GPU,
          names=names)
    elif self.config.regressor == "rotdim":
      names += ["target_rbox_angles", "target_rbox_dim"]
      outputs = utils.batch_slice(
          inputs,
          lambda a, b, c, d, e: detection_targets_graph(self.config, a, b, c, gt_rboxes=d, gt_angles=e),
          self.config.IMAGES_PER_GPU,
          names=names)
    elif self.config.regressor == "verts":
      names += ["target_rbox_verts"]
      outputs = utils.batch_slice(
          inputs,
          lambda a, b, c, d: detection_targets_graph(self.config, a, b, c, gt_rboxes=d),
          self.config.IMAGES_PER_GPU,
          names=names)
    else:
      outputs = utils.batch_slice(
          inputs,
          lambda a, b, c: detection_targets_graph(self.config, a, b, c),
          self.config.IMAGES_PER_GPU,
          names=names)

    return outputs

  def compute_output_shape(self, input_shape):
    output_shape = [
        (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # rois
        (None, 1),  # class_ids
        (None, self.config.TRAIN_ROIS_PER_IMAGE, 4),  # deltas
    ]

    if self.config.regressor == "deltas":
      output_shape += [(None, self.config.TRAIN_ROIS_PER_IMAGE,
                        4)]  # RBOX topline deltas
    elif self.config.regressor == "rotdim":
      output_shape += [
          (None, self.config.TRAIN_ROIS_PER_IMAGE, 1),  # RBOX angle
          (None, self.config.TRAIN_ROIS_PER_IMAGE, 2)
      ]  # RBOX dimensions
    elif self.config.regressor == "verts":
      output_shape += [(None, self.config.TRAIN_ROIS_PER_IMAGE,
                        8)]  # RBOX vertices
    return output_shape

  def compute_mask(self, inputs, mask=None):
    mask = [None, None, None]

    if self.config.regressor == "deltas" or self.config.regressor == "verts":
      mask += [None]
    elif self.config.regressor == "rotdim":
      mask += [None, None]

    return mask


# ############################################################
# #  DetectionLayer Class
# ############################################################
class DetectionLayer(KE.Layer):
  def __init__(self, config=None, **kwargs):
    super(DetectionLayer, self).__init__(**kwargs)
    self.config = config

  def call(self, inputs):

    rois = inputs[0]
    rbox_class = inputs[1]
    rbox_bbox = inputs[2]
    image_meta = inputs[3]

    _, _, window, _ = model.parse_image_meta_graph(image_meta)

    if self.config.regressor == "deltas":
      rbox_deltas = inputs[4]
      detections_batch = utils.batch_slice(
          [rois, rbox_class, rbox_bbox, window, rbox_deltas],
          lambda a, b, c, d, e: refine_detections_graph(a, b, c, d, self.config, rbox_deltas=e),
          self.config.IMAGES_PER_GPU)
      output_size = 10
    elif self.config.regressor == "rotdim":
      rbox_angles = inputs[4]
      rbox_dim = inputs[5]
      detections_batch = utils.batch_slice(
          [rois, rbox_class, rbox_bbox, window, rbox_angles, rbox_dim],
          lambda a, b, c, d, e, f: refine_detections_graph(a, b, c, d, self.config, rbox_angles=e, rbox_dim=f),
          self.config.IMAGES_PER_GPU)
      output_size = 9
    elif self.config.regressor == "verts":
      rbox_verts = inputs[4]
      detections_batch = utils.batch_slice(
          [rois, rbox_class, rbox_bbox, window, rbox_verts],
          lambda a, b, c, d, e: refine_detections_graph(a, b, c, d, self.config, rbox_verts=e),
          self.config.IMAGES_PER_GPU)
      output_size = 14
    else:
      detections_batch = utils.batch_slice(
          [rois, rbox_class, rbox_bbox, window],
          lambda a, b, c, d: refine_detections_graph(a, b, c, d, self.config),
          self.config.IMAGES_PER_GPU)
      output_size = 6

    return tf.reshape(detections_batch, [
        self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES,
        output_size
    ])

  def compute_output_shape(self, input_shape):

    if self.config.regressor == "deltas":
      return (None, self.config.DETECTION_MAX_INSTANCES, 10)
    elif self.config.regressor == "rotdim":
      return (None, self.config.DETECTION_MAX_INSTANCES, 9)
    elif self.config.regressor == "verts":
      return (None, self.config.DETECTION_MAX_INSTANCES, 14)

    return (None, self.config.DETECTION_MAX_INSTANCES, 6)


############################################################
#  Base Class
############################################################
class Base(object):
  def __init__(self, keras_model, config):
    self.config = config
    self.model_dir = os.path.join(os.getcwd(), "logs")
    self.set_log_dir()
    self.keras_model = keras_model

  def set_log_dir(self, model_path=None):
    # Set date and epoch counter as if starting a new model
    self.epoch = 0
    now = datetime.datetime.now()
    # If we have a model path with date and epochs use them
    if model_path:
      regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/rboxnet\_\w+(\d{4})\.h5"
      m = re.match(regex, model_path)
      if m:
        now = datetime.datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
            int(m.group(5)))
        self.epoch = int(m.group(6)) + 1

    # Directory for training logs
    self.log_dir = os.path.join(
        self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(),
                                                  now))

    # Path to save after each epoch. Include placeholders that get filled by Keras.
    self.checkpoint_path = os.path.join(
        self.log_dir, "rboxnet_{}_*epoch*.h5".format(self.config.NAME.lower()))
    self.checkpoint_path = self.checkpoint_path.replace(
        "*epoch*", "{epoch:04d}")

  def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
    # Print message on the first call (but not on recursive calls)
    if verbose > 0 and keras_model is None:
      log("Selecting layers to train")

    keras_model = keras_model or self.keras_model

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = keras_model.inner_model.layers if hasattr(
        keras_model, "inner_model") else keras_model.layers

    for layer in layers:
      # Is the layer a model?
      if layer.__class__.__name__ == 'Model':
        print("In model: ", layer.name)
        self.set_trainable(layer_regex, keras_model=layer, indent=indent + 4)
        continue

      if not layer.weights:
        continue

      # Is it trainable?
      trainable = bool(fullmatch(layer_regex, layer.name))
      # Update layer. If layer is a container, update inner layer.
      if layer.__class__.__name__ == 'TimeDistributed':
        layer.layer.trainable = trainable
      else:
        layer.trainable = trainable
      # Print trainble layer names
      if trainable and verbose > 0:
        log("{}{:20}   ({})".format(" " * indent, layer.name,
                                    layer.__class__.__name__))

  def load_weights(self, filepath, by_name=False, exclude=None):
    import h5py
    from keras.engine import saving

    if exclude:
      by_name = True

    if h5py is None:
      raise ImportError('`load_weights` requires h5py.')

    f = h5py.File(filepath, mode='r')
    if 'layer_names' not in f.attrs and 'model_weights' in f:
      f = f['model_weights']

    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    keras_model = self.keras_model
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
        else keras_model.layers

    # Exclude some layers
    if exclude:
      layers = filter(lambda l: l.name not in exclude, layers)

    if by_name:
      saving.load_weights_from_hdf5_group_by_name(f, layers)
    else:
      saving.load_weights_from_hdf5_group(f, layers)
    if hasattr(f, 'close'):
      f.close()

    # Update the log directory
    self.set_log_dir(filepath)

  def find_last(self):
    # Get directory names. Each directory corresponds to a model
    dir_names = next(os.walk(self.model_dir))[1]
    key = self.config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
      return None, None
    # Pick last directory
    dir_name = os.path.join(self.model_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("rboxnet"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
      return dir_name, None
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return dir_name, checkpoint

  def mold_inputs(self, images):
    molded_images = []
    image_metas = []
    windows = []
    for image in images:
      # Resize image to fit the model expected size
      # TODO: move resizing to mold_image()
      molded_image, window, scale, padding = \
          utils.resize_image(
              image,
              min_dim=self.config.IMAGE_MIN_DIM,
              max_dim=self.config.IMAGE_MAX_DIM,
              padding=self.config.IMAGE_PADDING)

      molded_image = model.mold_image(molded_image, self.config)

      # Build image_meta
      image_meta = model.compose_image_meta(
          0, image.shape, window,
          np.zeros([self.config.NUM_CLASSES], dtype=np.int32))

      # Append
      molded_images.append(molded_image)
      windows.append(window)
      image_metas.append(image_meta)

    # Pack into arrays
    molded_images = np.stack(molded_images)
    image_metas = np.stack(image_metas)
    windows = np.stack(windows)
    return molded_images, image_metas, windows
