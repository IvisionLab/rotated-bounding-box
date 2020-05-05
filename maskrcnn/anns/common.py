import os
import math
import csv
import numpy as np


def get_annotation_path(filepath, suffix=".txt"):
  path, filename = os.path.split(filepath)
  base_name = os.path.splitext(filename)[0]
  return os.path.join(path, base_name + suffix)


def load_yolo_annotation(filepath):
  annotation_path = get_annotation_path(filepath)
  with open(annotation_path, 'r') as txtfile:
    id, cx, cy, w, h = txtfile.readline().rstrip().split(' ')
    return int(id), np.array([cx, cy, w, h], dtype=np.float32)


def load_yolo_rot_annotation(filepath, suffix="-rot.txt"):
  annotation_path = get_annotation_path(filepath, suffix)
  with open(annotation_path, 'r') as txtfile:
    id, cx, cy, w, h, t = txtfile.readline().rstrip().split(' ')
    return int(id), np.array([cx, cy, w, h, t], dtype=np.float32)


def get_rbbox_annotation(filepath, suffix=".txt"):
  with open(get_annotation_path(filepath, suffix), 'r') as txtfile:
    id, x1, y1, x2, y2, cx, cy, w, h, t = txtfile.readline().rstrip().split(
        ' ')
    return int(id), np.array([x1, y1, x2, y2, cx, cy, w, h, t],
                             dtype=np.float32)


def rescale_box(size, box):
  for i in range(len(box)):
    box[i] = box[i] * size[i % 2]
  return box


def box2vert(box):
  w2 = box[2] * 0.5
  h2 = box[3] * 0.5
  return np.array(
      np.round([box[0] - w2, box[1] - h2, box[0] + w2, box[1] + h2]),
      dtype=np.int32)


def rbox2vert(rbox, size=(1, 1)):
  scale_factor = np.amax(size)
  return rbox2vert(rbox) * scale_factor


def rbox2points(rb):
  verts = rbox2vert(rb)
  pts = vert2points(verts)
  return pts


def rbox2vert(rbox):
  cx, cy, w, h, t = rbox
  angle = t * math.pi / 180.0
  cosine = math.cos(angle)
  sine = math.sin(angle)

  w2 = w * 0.5
  h2 = h * 0.5

  x1 = cx - w2 * cosine + h2 * sine
  y1 = cy - w2 * sine - h2 * cosine

  x2 = cx + w2 * cosine + h2 * sine
  y2 = cy + w2 * sine - h2 * cosine

  x3 = cx + w2 * cosine - h2 * sine
  y3 = cy + w2 * sine + h2 * cosine

  x4 = cx - w2 * cosine - h2 * sine
  y4 = cy - w2 * sine + h2 * cosine
  return np.array([x1, y1, x2, y2, x3, y3, x4, y4], dtype=np.float32)


def vert2points(v):
  pts = []
  for i in range(0, len(v), 2):
    pts.append((v[i], v[i + 1]))
  return np.array(pts)


def load_bbox_annotations(files, n=None):
  y = np.array([]).reshape(0, 6)
  for f in files:
    rows = []
    with open(f, 'rb') as csvfile:
      spamreader = csv.reader(csvfile)
      for row in spamreader:
        rows.append(row)
    x = np.array(rows)
    np.random.shuffle(x)
    if not n == None:
      y = np.concatenate([y, x[0:n]])
    else:
      y = np.concatenate([y, x])

  np.random.shuffle(y)
  return y


def parse_bbox_annotations(anns):
  elems = []
  for ann in anns:
    e = {}
    e["imgfile"] = ann[0]
    e["class_name"] = ann[5]
    e["bbox"] = ann[1:5].astype(float)
    elems += [e]
  return np.array(elems)


def get_ground_truth(img_path):
  gt_id, gt = get_rbbox_annotation(img_path)
  gt_box = gt[:4]
  gt_rbox = gt[4:]
  gt_rbox[0:2] += gt_box[0:2]
  return gt_id, gt_box, gt_rbox

def list_files(base_folder, limit_per_cls):
  import fnmatch
  import random

  clsid_cnt = {}
  all_anns = []
  for root, dirs, files in os.walk(base_folder):
    files = fnmatch.filter(files, '*.png')
    files = fnmatch.filter(files, '*[!-mask].png')
    files = sorted(files)

    for name in files:
      img_path = os.path.join(root, name)
      file_name = img_path.replace(base_folder, '')
      clsid, gt = get_rbbox_annotation(img_path)

      if limit_per_cls > 0 and \
          clsid in clsid_cnt and \
          clsid_cnt[clsid] >= limit_per_cls:
        break

      if clsid in clsid_cnt:
        clsid_cnt[clsid] += 1
      else:
        clsid_cnt[clsid] = 1
      all_anns.append({'file_name': file_name, 'clsid': clsid, 'gt': gt})

  return all_anns

def refine_rbox(rbox, bbox):
  x1, y1, x2, y2, x3, y3 = rbox[:6]
  cx = bbox[0] + (bbox[2] - bbox[0]) * 0.5
  cy = bbox[1] + (bbox[3] - bbox[1]) * 0.5
  rw = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
  rh = math.sqrt(math.pow(x3 - x2, 2) + math.pow(y3 - y2, 2))
  dx = x2-x1
  dy = y2-y1
  rad = math.atan2(dy, dx)
  t = math.degrees(rad)
  rbox = rbox2vert([cx, cy, rw, rh, t])
  return rbox.tolist()

def adjust_rotated_boxes(rbox, bbox):
  rbox = np.array(rbox)
  bbox = np.array(bbox)

  tolerance = 5
  dy = rbox[1]-rbox[3]
  if abs(dy) > tolerance:
    if dy < 0:
      dy1 = abs(bbox[1]-rbox[1])
      if (dy1 > tolerance):
        rbox[1]=bbox[1]
        rbox[5]=bbox[3]-(rbox[1]-bbox[1])
        rbox = refine_rbox(rbox, bbox)
    else:
      dy2 = abs(rbox[3]-bbox[3])
      if (dy2 > tolerance):
        rbox[3]=bbox[1]
        rbox[7]=bbox[1]+(bbox[3]-rbox[3])
        rbox = refine_rbox(rbox, bbox)

  return rbox

def rescale_rbox(rbox):
  rbox = np.array(rbox)
  for i in range(4):
    rbox[i] = rbox[i] + 6

  for i in range(4,8):
    rbox[i] = rbox[i] - 6
  return rbox