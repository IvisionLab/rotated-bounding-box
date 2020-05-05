import os
import json
import numpy as np
import random
from maskrcnn import utils
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

class CocoDataset(utils.Dataset):
    def load_coco(self, anns_file):
        coco = COCO(anns_file)

        # All classes
        class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        random.shuffle(image_ids)

        # Add classes
        for i in class_ids:
            self.add_class("gemini", i, coco.loadCats(i)[0]["name"])
        print(self.class_info)

        # Add images
        for i in image_ids:
            self.add_image(
                "gemini",
                image_id=i,
                path=coco.imgs[i]['coco_url'],
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "gemini":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "gemini.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "gemini":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

class MultibeamDataset(utils.Dataset):
    LABELS = ["ssiv_bahia", "jequitaia", "balsa"]
    def load_multibeam(self, dataset, base_folder, labels=None):
        for i in range(len(MultibeamDataset.LABELS)):
            self.add_class('gemini', i, MultibeamDataset.LABELS[i])

        labels_count = np.zeros(len(self.LABELS), dtype=np.int32)

        for item in dataset:
            if labels:
              seen = False
              for a in item['annotations']:
                label_id = a["id"]
                if self.LABELS[label_id] in labels:
                  seen = True
                  break

              if not seen:
                continue

            if not base_folder:
              image_path = os.path.join(item['basefolder'], item['filepath'])
            else:
              image_path = os.path.join(base_folder, item['filepath'])

            self.add_image(
                "gemini",
                image_id=item['filepath'],
                path=image_path,
                width=item['width'],
                height=item['height'],
                annotations=item['annotations'])

            for a in item['annotations']:
                labels_count[a['id']] += 1

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "gemini":
            return super(MultibeamDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        image_info = self.image_info[image_id]
        for a in image_info['annotations']:
            h = image_info["height"]
            w = image_info["width"]
            m = self.annToMask(a, h, w)
            instance_masks.append(m)
            class_id = self.map_source_class_id("gemini.{}".format(a['id']))
            class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(MultibeamDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "gemini":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def annToRLE(self, ann, height, width):
        segm = ann['segmentation']
        if isinstance(segm, list):
            rles = maskUtils.frPyObjects([segm], height, width)
            rle = maskUtils.merge(rles)
            return rle
        return None

    def annToMask(self, ann, height, width):
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def load_splitted(filepath):
  dataset = load_json(filepath, shuffle=True)
  n = int(0.8*len(dataset))

  dataset_train = MultibeamDataset()
  dataset_train.load_multibeam(dataset[:n])
  dataset_train.prepare()

  dataset_val = MultibeamDataset()
  dataset_val.load_multibeam(dataset[n:])
  dataset_val.prepare()

  return dataset_train, dataset_val

def load_json(filepath, shuffle=False):
  with open(filepath) as f:
    annotations = json.load(f)
    if shuffle:
      random.shuffle(annotations)
    return annotations

def gemini_dataset(anns, shuffle=True, base_folder=None, labels=None):
  dataset = MultibeamDataset()
  dataset.load_multibeam(load_json(anns, shuffle=shuffle), base_folder=base_folder, labels=labels)
  dataset.prepare()
  return dataset

def gemini_training_dataset(train_annotations, valid_annotations=None, base_folder=None):
  if valid_annotations == None:
    return load_splitted(train_annotations)

  dataset_train = gemini_dataset(train_annotations, base_folder=base_folder)
  dataset_val = gemini_dataset(valid_annotations, base_folder=base_folder)
  return dataset_train, dataset_val
