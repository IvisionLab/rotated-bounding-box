import cv2
import numpy as np
import math
from PIL import ImageFont, ImageDraw, Image
from maskrcnn.anns.common import vert2points

def draw_points(im, pts, colors=(255, 0, 0)):
  n = len(pts)
  for i in range(n):
    pt1 = tuple(np.round(pts[i % n]).astype(int))
    pt2 = tuple(np.round(pts[(i + 1) % n]).astype(int))
    cv2.line(im, pt1, pt2, colors, 3, lineType=cv2.LINE_AA)

def draw_verts(im, verts, colors=(255, 0, 0)):
  draw_points(im, vert2points(verts), colors)

def draw_rotated_boxes(im, rboxes_verts, colors=(255, 0, 0)):
    for verts in rboxes_verts:
      draw_verts(im, verts, colors)

def rotate(center, point, angle):
  rad = math.radians(angle)
  ox, oy = center
  px, py = point
  qx = ox + math.cos(rad) * (px - ox) - math.sin(rad) * (py - oy)
  qy = oy + math.sin(rad) * (px - ox) + math.cos(rad) * (py - oy)
  return (qx, qy)

def draw_rotate(im, box, angle, xy=(0,0), color=(0, 0, 0)):
  w, h = box.size
  cx = w / 2
  cy = h / 2
  xx = []
  yy = []
  for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
    x, y = rotate((cx, cy), (x, y), -angle)
    xx.append(x)
    yy.append(y)
  nw = int(math.ceil(max(xx)) - math.floor(min(xx)))
  nh = int(math.ceil(max(yy)) - math.floor(min(yy)))
  sw = (nw - w) / 2.0
  sh = (nh - h) / 2.0

  xx = np.array(xx)
  yy = np.array(yy)
  xx = (xx + sw).astype(int)
  yy = (yy + sh).astype(int)

  box = box.rotate(angle, resample=Image.BILINEAR, expand=1)
  sx, sy = box.size
  # px = int(round(xy[0]-xx[3]))
  # py = int(round(xy[1]-yy[3]))
  px = int(round(xy[0]-xx[3]))
  py = int(round(xy[1]-yy[3]))
  im.paste(box, (px, py, px+sx, py+sy), box)
  return im

def draw_detection_info(image, text, xy=(0, 0), color=(0, 255, 0), angle=None):
  x1, y1 = xy
  im_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
  pil_im = Image.fromarray(im_rgb)

  font = ImageFont.truetype("arial-bold.ttf", 30)

  text_size = font.getsize(text)
  box_size = (text_size[0]+4, text_size[1]+12)
  box=Image.new('RGBA', box_size)
  draw = ImageDraw.Draw(box)
  box_coords = [0, 0, box_size[0], box_size[1]]
  draw.rectangle(box_coords, fill=color)
  draw.text((2, 6), text, font=font, fill=(255,255,255,255))

  if angle is None:
    sx, sy = box.size
    px = int(round(xy[0]))
    py = int(round(xy[1]-sy))
    pil_im.paste(box, (px, py, px+sx, py+sy), box)
  else:
    draw_rotate(pil_im, box, angle, xy=xy, color=color)
  image = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
  return image


def draw_boxes(image, boxes, color=(0, 255, 0), class_ids=None, scores=None, labels=None, ious=None):
  image_h, image_w, _ = image.shape

  for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

    if not labels is None:
      label =  labels[class_ids[i]] if not class_ids is None else "None"

      if not scores is None and not ious is None:
        text = "{0} (Score={1:.2f}, IoU={2:.2f})".format(label, scores[i], ious[i])
      elif not scores is None:
        text = "{0} (Score={1:.2f})".format(label, scores[i])
      elif not ious is None:
        text = "{0} (IoU={1:.2f})".format(label, ious[i])
      else:
        text = "{0}".format(label)


      if y1 < 50:
        image = draw_detection_info(image, text, xy=(x1, y2+45), color=color[::-1])
      else:
        image = draw_detection_info(image, text, xy=(x1, y1), color=color[::-1])

  return image


def draw_rotated_detections(image, detections, color=(0, 0, 255), labels=None, ious=None, score=False):
  for i, dt in enumerate(detections):
    verts = dt['rbox']
    draw_verts(image, verts, colors=color)
    if not labels is None:
      x1, y1, x2, y2, x3, y3, x4, y4 = verts[0], verts[1], verts[2], verts[3], verts[4], verts[5], verts[6], verts[7]
      dx = x2-x1
      dy = y2-y1
      rad = math.atan2(dy, dx)
      angle = math.degrees(rad)

      label =  labels[dt['id']]

      if score and not ious is None:
        text = "{0} (Score={1:.2f}, IoU={2:.2f})".format(label, dt['score'], ious[i])
      elif score:
        text = "{0} (Score={1:.2f})".format(label, dt['score'])
      elif not ious is None:
        text = "{0} (IoU={1:.2f})".format(label, ious[i])
      else:
        text = "{0}".format(label)

      if ious[i] <= 0.01:
        image = draw_detection_info(image, text, xy=(x1, y1), color=color[::-1], angle=-angle)
      else:
        image = draw_detection_info(image, text, xy=(x1, y1), color=color[::-1], angle=-angle)


  return image

