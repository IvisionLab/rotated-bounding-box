import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.models as KM
from maskrcnn import model, base, utils
from maskrcnn.model import log
from maskrcnn.generator import data_generator


class Training(base.Base):
  def __init__(self, config):
    super().__init__(self.build(config), config)

  def build(self, config):
    # Check image size
    base.check_image_size(config)

    # Input layers
    inputs = self.input_layers(config)
    input_image = inputs[0]
    input_image_meta = inputs[1]
    input_rpn_match = inputs[2]
    input_rpn_bbox = inputs[3]
    input_gt_class_ids = inputs[4]
    input_gt_boxes = inputs[5]
    input_gt_rboxes = inputs[6]
    input_gt_angles = inputs[7]

    # Class ID mask to mark class IDs supported by the dataset the image came from.
    _, _, _, active_class_ids = KL.Lambda(
        lambda x: model.parse_image_meta_graph(x),
        mask=[None, None, None, None])(input_image_meta)

    # Normalize bounding-box coordinates
    gt_boxes = self.box_normalize_layer(input_image, input_gt_boxes)

    gt_rboxes = None
    if self.is_valid_regressor(config.regressor):
      # Normalize rotated bounding-box coordinates
      gt_rboxes = self.rbox_normalize_layer(input_image, input_gt_rboxes)

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
        rpn_feature_maps, self.anchors, config.POST_NMS_ROIS_TRAINING,
        config)

    # RPN class loss
    rpn_class_loss = \
        base.rpn_class_loss_layer(input_rpn_match, rpn_class_logits)

    # RPN bbox loss
    rpn_bbox_loss = \
        base.rpn_bbox_loss_layer(config, input_rpn_bbox, input_rpn_match, rpn_bbox)

    if config.regressor == "deltas":
      # Generate detection targets with rotated bounding-box using top line deltas
      rois, target_class_ids, target_bbox, target_rbox_deltas = \
          base.detection_target_layer(config,
            [rpn_rois, input_gt_class_ids, gt_boxes, gt_rboxes])
    elif config.regressor == "rotdim":
      # Generate detection targets with rotated bounding-box using angles and dimension
      rois, target_class_ids, target_bbox, target_rbox_angles, target_rbox_dim = \
          base.detection_target_layer(config,
            [rpn_rois, input_gt_class_ids, gt_boxes, gt_rboxes, input_gt_angles])
    elif config.regressor == "verts":
      # Generate detection targets with rotated bounding-box using vertices
      rois, target_class_ids, target_bbox, target_rbox_verts = \
          base.detection_target_layer(config,
            [rpn_rois, input_gt_class_ids, gt_boxes, gt_rboxes])
    else:
      # Generate detection targets
      rois, target_class_ids, target_bbox = \
          base.detection_target_layer(config, [rpn_rois, input_gt_class_ids, gt_boxes])

    # ROI Align layer
    roi_align = base.roi_align_layers(rois, rbox_feature_maps, config)
    # Two 1024 FC layers
    fc_layers = base.full_connected_layers(roi_align, config)

    # Bounding box classifier layers
    rbox_class_logits, rbox_class, rbox_bbox = \
        base.bbox_classifier_layers(fc_layers, config.NUM_CLASSES)

    # Output ROIs
    output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

    # RBOX class loss
    rbox_class_loss = \
        base.rbox_class_loss_layer(target_class_ids, rbox_class_logits, active_class_ids)

    # RBOX bbox loss
    rbox_bbox_loss = \
        base.rbox_bbox_loss_layer(target_bbox, target_class_ids, rbox_bbox)

    # Training outputs
    outputs = \
        [rpn_class_logits, rpn_class, rpn_bbox, rpn_rois,
         rbox_class_logits, rbox_class, rbox_bbox, output_rois]

    # Loss functions outputs
    outputs_loss = \
        [rpn_class_loss, rpn_bbox_loss,
         rbox_class_loss, rbox_bbox_loss]

    if config.regressor == "deltas":
      # RBOX topline deltas regressor layers
      rbox_deltas = \
          base.rbox_deltas_regressor_layers(fc_layers, config.NUM_CLASSES)
      # Add RBOX topline deltas to outputs
      outputs += [rbox_deltas]
      # RBOX topline deltas loss
      rbox_deltas_loss = \
          base.rbox_deltas_loss_layer(target_rbox_deltas, target_class_ids, rbox_deltas)
      # Add RBOX topline deltas loss to outputs
      outputs_loss += [rbox_deltas_loss]
    elif config.regressor == "rotdim":
      # RBOX angles+dimension regressor layers
      rbox_angles, rbox_dim = \
          base.rbox_rotdim_regressor_layers(fc_layers, config.NUM_CLASSES)
      # Add RBOX angles and dimensions to outputs
      outputs += [rbox_angles, rbox_dim]
      # RBOX angles loss
      rbox_angles_loss = \
          base.rbox_angles_loss_layer(target_rbox_angles, target_class_ids, rbox_angles)
      # RBOX angles loss
      rbox_dim_loss = \
          base.rbox_dim_loss_layer(target_rbox_dim, target_class_ids, rbox_dim)
      outputs_loss += [rbox_angles_loss, rbox_dim_loss]
    elif config.regressor == "verts":
      # RBOX vertices regressor layers
      rbox_verts = \
          base.rbox_verts_regressor_layers(fc_layers, config.NUM_CLASSES)
      # Add RBOX vertices to outputs
      outputs += [rbox_verts]
      # RBOX vertices loss
      rbox_verts_loss = \
          base.rbox_verts_loss_layer(target_rbox_verts, target_class_ids, rbox_verts)
      # Add RBOX topline deltas loss to outputs
      outputs_loss += [rbox_verts_loss]

    outputs += outputs_loss

    return KM.Model(inputs, outputs, name='rboxnet')

  def train(self, train_dataset, val_dataset, learning_rate, epochs, layers):
    # Pre-defined layer regular expressions
    layer_regex = {
        # all layers but the backbone
        "heads": r"(rbox\_.*)|(rpn\_.*)|(fpn\_.*)",
        # From a specific Resnet stage and up
        "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rbox\_.*)|(rpn\_.*)|(fpn\_.*)",
        "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rbox\_.*)|(rpn\_.*)|(fpn\_.*)",
        "5+": r"(res5.*)|(bn5.*)|(rbox\_.*)|(rpn\_.*)|(fpn\_.*)",
        # All layers
        "all":
        ".*",
    }

    if layers in layer_regex.keys():
      layers = layer_regex[layers]

    # Data generators
    train_generator = data_generator(
        train_dataset,
        self.config,
        shuffle=True,
        batch_size=self.config.BATCH_SIZE,
        augment=False)

    val_generator = data_generator(
        val_dataset,
        self.config,
        shuffle=True,
        batch_size=self.config.BATCH_SIZE,
        augment=False)

    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=False),
        keras.callbacks.ModelCheckpoint(
            self.checkpoint_path, verbose=1, save_weights_only=True),
    ]

    # Train
    log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
    log("Checkpoint Path: {}".format(self.checkpoint_path))
    self.set_trainable(layers)
    self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

    workers = max(self.config.BATCH_SIZE // 2, 2)
    self.keras_model.fit_generator(
        train_generator,
        verbose=1,
        initial_epoch=self.epoch,
        epochs=epochs,
        steps_per_epoch=self.config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=next(val_generator),
        validation_steps=self.config.VALIDATION_STEPS,
        max_queue_size=100,
        workers=workers,
        use_multiprocessing=True)

    self.epoch = max(self.epoch, epochs)

  def compile(self, learning_rate, momentum):
    # Optimizer object
    optimizer = keras.optimizers.SGD(
        lr=learning_rate, momentum=momentum, clipnorm=5.0)

    # Add Losses
    # First, clear previously set losses to avoid duplication
    self.keras_model._losses = []
    self.keras_model._per_input_losses = {}

    loss_names = [
        "rpn_class_loss", "rpn_bbox_loss", "rbox_class_loss", "rbox_bbox_loss"
    ]

    if self.config.regressor == "deltas":
      loss_names += ["rbox_deltas_loss"]
    elif self.config.regressor == "rotdim":
      loss_names += ["rbox_angles_loss", "rbox_dim_loss"]
    elif self.config.regressor == "verts":
      loss_names += ["rbox_verts_loss"]

    for name in loss_names:
      layer = self.keras_model.get_layer(name)
      if layer.output in self.keras_model.losses:
        continue
      self.keras_model.add_loss(tf.reduce_mean(layer.output, keep_dims=True))

    # Add L2 Regularization
    # Skip gamma and beta weights of batch normalization layers.
    reg_losses = [
        keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(
            tf.size(w), tf.float32) for w in self.keras_model.trainable_weights
        if 'gamma' not in w.name and 'beta' not in w.name
    ]

    self.keras_model.add_loss(tf.add_n(reg_losses))

    # Compile
    self.keras_model.compile(
        optimizer=optimizer, loss=[None] * len(self.keras_model.outputs))

    # Add metrics for losses
    for name in loss_names:
      if name in self.keras_model.metrics_names:
        continue
      layer = self.keras_model.get_layer(name)
      self.keras_model.metrics_names.append(name)
      self.keras_model.metrics_tensors.append(
          tf.reduce_mean(layer.output, keep_dims=True))

  def input_layers(self, config):
    # input image layer
    input_image = KL.Input(
        shape=config.IMAGE_SHAPE.tolist(), name="input_image")

    # input image meta
    input_image_meta = KL.Input(shape=[None], name="input_image_meta")

    # input rpn match
    input_rpn_match = KL.Input(
        shape=[None, 1], name="input_rpn_match", dtype=tf.int32)

    # input rpn bbox
    input_rpn_bbox = KL.Input(
        shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

    # Detection GT (class IDs, bounding boxes, and masks)
    # GT Class IDs (zero padded)
    input_gt_class_ids = KL.Input(
        shape=[None], name="input_gt_class_ids", dtype=tf.int32)

    # GT Boxes in pixels (zero padded)
    # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
    input_gt_boxes = KL.Input(
        shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

    # GT rotated bounding box
    input_gt_rboxes = KL.Input(
        shape=[None, 8], name="input_gt_rboxes", dtype=tf.float32)

    # GT angles
    input_gt_angles = KL.Input(
        shape=[None, 1], name="input_gt_angles", dtype=tf.float32)

    return [
        input_image, input_image_meta, input_rpn_match, input_rpn_bbox,
        input_gt_class_ids, input_gt_boxes, input_gt_rboxes, input_gt_angles
    ]

  def box_normalize_layer(self, input_image, input_gt_boxes):
    h, w = K.shape(input_image)[1], K.shape(input_image)[2]
    image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
    return KL.Lambda(lambda x: x / image_scale)(input_gt_boxes)

  def rbox_normalize_layer(self, input_image, input_gt_rboxes):
    h, w = K.shape(input_image)[1], K.shape(input_image)[2]
    image_scale = K.cast(K.stack([h, w, h, w, h, w, h, w], axis=0), tf.float32)
    return KL.Lambda(lambda x: x / image_scale)(input_gt_rboxes)

  def is_valid_regressor(self, rbox_regr):
    return rbox_regr == "deltas" or rbox_regr == "rotdim" or rbox_regr == "verts"
