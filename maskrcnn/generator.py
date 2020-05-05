import logging
import numpy as np
import maskrcnn
from maskrcnn import utils
from maskrcnn.model import load_image_gt, build_rpn_targets, generate_random_rois, mold_image

def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0, batch_size=1, detection_targets=False):
  b = 0  # batch item index
  image_index = -1
  image_ids = np.copy(dataset.image_ids)
  error_count = 0

  # Anchors
  # [anchor_count, (y1, x1, y2, x2)]
  anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                           config.RPN_ANCHOR_RATIOS,
                                           config.BACKBONE_SHAPES,
                                           config.BACKBONE_STRIDES,
                                           config.RPN_ANCHOR_STRIDE)

  # Keras requires a generator to run indefinately.
  while True:
    try:
      # Increment index to pick next image. Shuffle if at the start of an epoch.
      image_index = (image_index + 1) % len(image_ids)
      if shuffle and image_index == 0:
        np.random.shuffle(image_ids)

      # Get GT bounding boxes and masks for image.
      image_id = image_ids[image_index]
      image_gt = load_image_gt(dataset, config, image_id, augment=augment, use_mini_mask=config.USE_MINI_MASK)
      image, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_rboxes, gt_angles = image_gt

      # Skip images that have no instances. This can happen in cases
      # where we train on a subset of classes and the image doesn't
      # have any of the classes we care about.
      if not np.any(gt_class_ids > 0):
        continue

      # RPN Targets
      rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                              gt_class_ids, gt_boxes, config)

      # Mask R-CNN Targets
      if random_rois:
        rpn_rois = generate_random_rois(image.shape, random_rois, gt_class_ids, gt_boxes)
        if detection_targets:
          dt_targets = build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)
          rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = dt_targets

      # Init batch arrays
      if b == 0:
        batch_image_meta = np.zeros((batch_size,) + image_meta.shape, dtype=image_meta.dtype)
        batch_rpn_match = np.zeros([batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
        batch_rpn_bbox = np.zeros([batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
        batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
        batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
        batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
        batch_gt_rboxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 8), dtype=np.int32)
        batch_gt_angles = np.zeros((batch_size, config.MAX_GT_INSTANCES, 1), dtype=np.float32)

        if config.USE_MINI_MASK:
          batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1], config.MAX_GT_INSTANCES))
        else:
          batch_gt_masks = np.zeros((batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))

        if random_rois:
          batch_rpn_rois = np.zeros((batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
          if detection_targets:
            batch_rois = np.zeros((batch_size,) + rois.shape, dtype=rois.dtype)
            batch_mrcnn_class_ids = np.zeros((batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
            batch_mrcnn_bbox = np.zeros((batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
            batch_mrcnn_mask = np.zeros((batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

      # If more instances than fits in the array, sub-sample from them.
      if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
        ids = np.random.choice(np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
        gt_class_ids = gt_class_ids[ids]
        gt_boxes = gt_boxes[ids]
        gt_rboxes = gt_rboxes[ids]
        gt_angles = gt_angles[ids]
        gt_masks = gt_masks[:, :, ids]

      # Add to batch
      batch_image_meta[b] = image_meta
      batch_rpn_match[b] = rpn_match[:, np.newaxis]
      batch_rpn_bbox[b] = rpn_bbox
      batch_images[b] = mold_image(image.astype(np.float32), config)
      batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
      batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
      batch_gt_rboxes[b,:gt_rboxes.shape[0]] = gt_rboxes
      batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
      batch_gt_angles[b,:gt_angles.shape[0]] = gt_angles
      if random_rois:
        batch_rpn_rois[b] = rpn_rois
        if detection_targets:
          batch_rois[b] = rois
          batch_mrcnn_class_ids[b] = mrcnn_class_ids
          batch_mrcnn_bbox[b] = mrcnn_bbox
          batch_mrcnn_mask[b] = mrcnn_mask
      b += 1

      # Batch full?
      if b >= batch_size:
        inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes, batch_gt_rboxes, batch_gt_angles]
        outputs = []

        if random_rois:
          inputs.extend([batch_rpn_rois])
          if detection_targets:
            inputs.extend([batch_rois])
            # Keras requires that output and targets have the same number of dimensions
            batch_mrcnn_class_ids = np.expand_dims(batch_mrcnn_class_ids, -1)
            outputs.extend([batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        yield inputs, outputs

        b = 0
        # start a new batch
    except (GeneratorExit, KeyboardInterrupt):
      raise
    except:
      # Log it and skip the image
      logging.exception("Error processing image {}".format(
          dataset.image_info[image_id]))
      error_count += 1
      if error_count > 5:
        raise