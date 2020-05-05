import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from maskrcnn import model, base, utils
from maskrcnn.model import log
from maskrcnn.base import unmold_detections, top_line_to_vertices, vertices_fliplr


class Inference(base.Base):
  def __init__(self, config):
    super().__init__(self.build(config), config)

  def build(self, config):
    # Check image size
    base.check_image_size(config)

    # input image layer
    input_image = KL.Input(
        shape=config.IMAGE_SHAPE.tolist(), name="input_image")

    input_image_meta = KL.Input(shape=[None], name="input_image_meta")

    # Feature extractor layer
    [P2, P3, P4, P5, P6] = base.feature_extractor_layers(input_image, config)
    rpn_feature_maps = [P2, P3, P4, P5, P6]
    rbox_feature_maps = [P2, P3, P4, P5]

    # Generate Anchors
    self.anchors = utils.generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS,
        config.BACKBONE_SHAPES, config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE)

    # RPN layer
    rpn_class_logits, rpn_class, rpn_bbox, rpn_rois = base.rpn_layers(
        rpn_feature_maps, self.anchors, config.POST_NMS_ROIS_INFERENCE, config)

    # ROI Align layer
    roi_align = base.roi_align_layers(rpn_rois, rbox_feature_maps, config)

    # Two 1024 FC layers
    fc_layers = base.full_connected_layers(roi_align, config)

    # Bounding box classifier layers
    rbox_class_logits, rbox_class, rbox_bbox = \
        base.bbox_classifier_layers(fc_layers, config.NUM_CLASSES)

    # Detection inputs
    detection_layer_inputs = [
        rpn_rois, rbox_class, rbox_bbox, input_image_meta
    ]

    if config.regressor == "deltas":
      # RBOX top line deltas regressor layers
      rbox_deltas = base.rbox_deltas_regressor_layers(fc_layers,
                                                      config.NUM_CLASSES)
      detection_layer_inputs += [rbox_deltas]
    elif config.regressor == "rotdim":
      # RBOX angles+dimension regressor layers
      rbox_angles, rbox_dim = \
          base.rbox_rotdim_regressor_layers(fc_layers, config.NUM_CLASSES)
      detection_layer_inputs += [rbox_angles, rbox_dim]
    elif config.regressor == "verts":
      # RBOX vertices regressor layers
      rbox_verts = \
          base.rbox_verts_regressor_layers(fc_layers, config.NUM_CLASSES)
      detection_layer_inputs += [rbox_verts]

    rbox_dts = \
      base.detection_layer(config, detection_layer_inputs)

    inputs = [input_image, input_image_meta]
    outputs = [rpn_rois, rpn_class, rpn_bbox, rbox_dts]

    return KM.Model(inputs, outputs, name='rboxnet')

  def detect(self, images, verbose=0):
    assert len(images) == self.config.BATCH_SIZE, \
        "len(images) must be equal to BATCH_SIZE"

    if verbose:
      log("Processing {} images".format(len(images)))
      for image in images:
        log("image", image)

    # Mold inputs to format expected by the neural network
    molded_images, image_metas, windows = self.mold_inputs(images)

    if verbose:
      log("molded_images", molded_images)
      log("image_metas", image_metas)

    rpn_rois, rpn_class, rpn_bbox, rbox_dts = self.keras_model.predict(
        [molded_images, image_metas], verbose=verbose)

    detections = []
    for i, image in enumerate(images):
      rotated_boxes = None
      if self.config.regressor == "deltas" or \
          self.config.regressor == "rotdim" or \
          self.config.regressor == "verts":
        class_ids, scores, boxes, rotated_boxes = unmold_detections(
            rbox_dts[i], image.shape, windows[i], self.config)
      else:
        class_ids, scores, boxes = unmold_detections(rbox_dts[i], image.shape,
                                                     windows[i], self.config)

      detections.append({
          "class_ids": class_ids,
          "scores": scores,
          "boxes": boxes,
          "rotated_boxes": rotated_boxes
      })

    return detections
