#!/usr/bin/env python3
import sys
import json
from maskrcnn import training, dataset, config


class TrainingConfig(config.Config):
  NAME = "gemini"
  GPU_COUNT = 1
  IMAGES_PER_GPU = 4
  NUM_CLASSES = 1 + 3
  IMAGE_MAX_DIM = 448
  BACKBONE = "resnet50"
  regressor = None

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='rboxnet training .')

  parser.add_argument(
      "config",
      metavar="/path/to/config/file",
      help="'Path to configuration file")

  parser.add_argument(
      '--model',
      required=False,
      metavar="/path/to/weights.h5",
      help="Path to weights .h5 file")

  parser.add_argument(
      '--dataset-base-folder',
      '-d',
      required=False,
      metavar="/path/to/dataset/base/folder",
      help="Path to dataset base folder")

  if len(sys.argv) > 1:
    args = parser.parse_args()

    anns_train = None
    anns_valid = None

    cfg = None
    with open(args.config) as f:
      cfg = json.load(f)
      anns_train = cfg['annotations']['train']
      anns_valid = cfg['annotations']['valid']
      dataset_train, dataset_valid = dataset.gemini_training_dataset(
          anns_train, anns_valid, args.dataset_base_folder)

    config = TrainingConfig()
    config.regressor = cfg['regressor']
    if config.regressor:
      config.NAME="{0}_{1}_{2}".format(config.NAME, config.BACKBONE, config.regressor)
      print("Configuration Name: ", config.NAME)

    net = training.Training(config)

    if args.model:
      if args.model.lower() == "last":
        model_path = net.find_last()[1]
      else:
        model_path = args.model
      print("Model: {0}".format(model_path))
      net.load_weights(model_path, by_name=True)

    print("Training network heads")
    net.train(
        dataset_train,
        dataset_valid,
        learning_rate=config.LEARNING_RATE,
        epochs=40,
        layers='heads')

    print("Fine tune Resnet stage 4 and up")
    net.train(
        dataset_train,
        dataset_valid,
        learning_rate=config.LEARNING_RATE,
        epochs=120,
        layers='4+')

    print("Fine tune all layers")
    net.train(
        dataset_train,
        dataset_valid,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=160,
        layers='all')
  else:
    print("Please inform the configuration")