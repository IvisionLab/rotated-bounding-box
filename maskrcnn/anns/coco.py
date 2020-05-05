import os
import cv2
import json
import numpy as np
from sklearn.utils import shuffle
from maskrcnn.anns import common
from maskrcnn.anns.rbox import load_rbbox, load_bbox, load_rbox_mask


################################
# Holder class
################################
class Holder:
  def __init__(self):
    self.coco = dict()
    self.coco['images'] = []
    self.coco['type'] = 'instances'
    self.coco['annotations'] = []
    self.coco['categories'] = []
    self.category_set = dict()
    self.image_set = set()
    self.category_item_id = 0
    self.annotation_id = 0
    self.image_id = 20180000000

  def addCatItem(self, name):
    category_item = dict()
    category_item['supercategory'] = 'none'
    self.category_item_id += 1
    category_item['id'] = self.category_item_id
    category_item['name'] = name
    self.coco['categories'].append(category_item)
    self.category_set[name] = self.category_item_id
    return self.category_item_id

  def addImgItem(self, file_name, size):
    if file_name is None:
      raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
      raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
      raise Exception('Could not find height tag in xml file.')
    self.image_id += 1
    image_item = dict()
    image_item['id'] = self.image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    self.coco['images'].append(image_item)
    self.image_set.add(file_name)
    return self.image_id

  def addAnnoItem(self, object_name, image_id, category_id, bbox, segm):
    annotation_item = dict()
    annotation_item['segmentation'] = []
    annotation_item['segmentation'].append(segm)
    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    self.annotation_id += 1
    annotation_item['id'] = self.annotation_id
    self.coco['annotations'].append(annotation_item)


def unpack_bbox(gt):
  gt = gt.astype(int)
  bbox = [int(gt[0]), int(gt[1]), int(gt[2] - gt[0]), int(gt[3] - gt[1])]

  return bbox


def unpack_rbbox(gt, bbox):
  rbbox = gt[4:]
  rbbox[0:2] += bbox[0:2]
  rbbox = common.rbox2points(rbbox).astype(int)
  polygon = []
  #left_top
  polygon.append(rbbox[0][0])
  polygon.append(rbbox[0][1])
  #left_bottom
  polygon.append(rbbox[1][0])
  polygon.append(rbbox[1][1])
  #right_bottom
  polygon.append(rbbox[2][0])
  polygon.append(rbbox[2][1])
  #right_top
  polygon.append(rbbox[3][0])
  polygon.append(rbbox[3][1])


def find_mask_polygon(mask_filepath):
  mask = cv2.imread(mask_filepath)
  mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
  _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
  return contours[0].flatten().tolist()


def get_mask_filepath(img_file_path):
  path, filename = os.path.split(img_file_path)
  base_name = os.path.splitext(filename)[0]
  return os.path.join(path, base_name + '-mask.png')


def build_coco_annotations(base_folder, all_anns, labels, use_rbbox):
  holder = Holder()
  for anns in all_anns:
    file_name = anns['file_name']
    clsid = anns['clsid']
    gt = anns['gt']
    size = {}


    img_file_path = os.path.join(base_folder, file_name)
    img = cv2.imread(img_file_path)
    size['height'], size['width'], size['depth'] = img.shape
    object_name = labels[clsid]

    if object_name not in holder.category_set:
      category_id = holder.addCatItem(object_name)
    else:
      category_id = holder.category_set[object_name]

    if file_name not in holder.image_set:
      image_id = holder.addImgItem(file_name, size)
      print('add image with {} and {}'.format(file_name, size))
    else:
      raise Exception('duplicated image: {}'.format(file_name))

    if use_rbbox:
      gt_rbbox = load_rbbox(gt)
      polygon, bbox = load_rbox_mask(gt_rbbox)
      bbox = list(bbox)
    else:
      bbox = unpack_bbox(gt.astype(np.int32))
      polygon = find_mask_polygon(get_mask_filepath(img_file_path))

    print('add annotation with {},{},{},{}'.format(
        object_name, image_id, category_id, bbox))
    holder.addAnnoItem(object_name, image_id, category_id,
                       bbox, polygon)

  return holder.coco

def generate(args):

  all_anns = common.list_files(args.base_folder, args.limit)
  coco_anns = build_coco_annotations(args.base_folder, all_anns,
                                     ["ssiv_bahia", "jequitaia", "balsa"],
                                     args.use_rbbox)

  json.dump(coco_anns, open("coco_annotations.json", 'w'))

  if args.split:
    shuffled_anns = shuffle(all_anns)
    n = int(0.8 * len(shuffled_anns))
    train_anns = shuffled_anns[:n]
    valid_anns = shuffled_anns[n:]
    coco_train_anns = build_coco_annotations(
        args.base_folder, train_anns, ["ssiv_bahia", "jequitaia", "balsa"],
        args.use_rbbox)

    json.dump(coco_anns, open("coco_annotations_train.json", 'w'))

    coco_valid_anns = build_coco_annotations(
        args.base_folder, valid_anns, ["ssiv_bahia", "jequitaia", "balsa"],
        args.use_rbbox)

    json.dump(coco_anns, open("coco_annotations_valid.json", 'w'))

def generate_from_dota(base_dir,
                       class_names,
                       prefix="dota",
                       suffix="test",
                       width=730,
                       height=422,
                       depth=3):
  class_to_index = dict(zip(class_names, range(len(class_names))))

  img_dir = os.path.join(base_dir, 'images')
  label_dir = os.path.join(base_dir, "labelTxt")

  dota_dataset = [None] * len(class_names)

  idx = 0
  filelist = os.listdir(img_dir)
  num_files = len(filelist)

  for filename in filelist:
    name = os.path.basename(os.path.splitext(filename)[0])
    img_path = os.path.join(img_dir, filename)
    label_path = os.path.join(label_dir, name+".txt")

    with open(label_path, 'r') as f:
      lines = f.readlines()
      lines = [line.strip().split(' ') for line in lines]

      objs = []
      for obj in lines:

        x1 = float(obj[0])
        y1 = float(obj[1])
        x2 = float(obj[2])
        y2 = float(obj[3])
        x3 = float(obj[4])
        y3 = float(obj[5])
        x4 = float(obj[6])
        y4 = float(obj[7])

        xmin = max(min(x1, x2, x3, x4), 0)
        xmax = max(x1, x2, x3, x4)
        ymin = max(min(y1, y2, y3, y4), 0)
        ymax = max(y1, y2, y3, y4)

        cls_id = class_to_index[obj[8].lower().strip()]

        obj = {
          "file_name": filename,
          "width": width,
          "height": height,
          "depth": depth,
          "cls_id": cls_id,
          "bbox": [xmin, ymin, xmax, ymax],
          "rbox": [x1, y1, x2, y2, x3, y3, x4, y4]
        }

        if dota_dataset[cls_id] is None:
          dota_dataset[cls_id] = []
        dota_dataset[cls_id] += [obj]
        idx+=1
        print("{}/{} {}".format(idx, num_files, filename))

  coco_anns = build_coco_annotations_from_dota(dota_dataset, class_names)

  coco_anns_filepath = '%s_coco_annotations_%s.json' % (prefix, suffix)
  json.dump(coco_anns, open(coco_anns_filepath, 'w'))

def build_coco_annotations_from_dota(dota_dataset, class_names):
  holder = Holder()
  idx = 0
  total = sum(map(len, dota_dataset))
  for annotations in dota_dataset:
    for anns in annotations:
      object_name = class_names[anns['cls_id']]

      file_name = anns['file_name']
      rbox = anns['rbox']
      bbox = anns["bbox"]

      size = {
        'height': anns['height'],
        'width': anns['width'],
        'depth': anns['depth']
      }

      if object_name not in holder.category_set:
        category_id = holder.addCatItem(object_name)
      else:
        category_id = holder.category_set[object_name]

      if file_name not in holder.image_set:
        image_id = holder.addImgItem(file_name, size)
        print('add image with {} and {}'.format(file_name, size))
      else:
        raise Exception('duplicated image: {}'.format(file_name))

      bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
      polygon = [int(x) for x in rbox]

      print('add annotation with {},{},{},{}'.format(object_name,
                                                    image_id,
                                                    category_id,
                                                    bbox))


      holder.addAnnoItem(object_name,
                        image_id,
                        category_id,
                        bbox,
                        polygon)
      idx += 1
      print("{}/{} {}".format(idx, total, file_name))

  return holder.coco