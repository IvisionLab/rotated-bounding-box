#!/usr/bin/python
import os
import cv2
import shutil
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.etree import ElementTree
from xml.dom import minidom
from maskrcnn.anns import common
from maskrcnn.anns.rbox import load_rbox_mask, load_rbbox


def prettify(elem):
  """Return a pretty-printed XML string for the Element.
    """
  rough_string = ElementTree.tostring(elem, 'utf-8')
  reparsed = minidom.parseString(rough_string)
  return reparsed.toprettyxml(indent="  ")


def savexml(elem, base_folder, output_folder, labels):

  filename = elem['file_name'][1:]

  img_path = os.path.join(base_folder, filename)
  img = cv2.imread(img_path)
  height, width, depth = img.shape
  class_name = labels[elem['clsid']]
  rbbox = load_rbbox(elem['gt'])
  _, bbox = load_rbox_mask(rbbox)

  root = ET.Element("annotation", verified="yes")
  ET.SubElement(root, "folder").text = base_folder
  ET.SubElement(root, "filename").text = filename
  ET.SubElement(root, "path").text = img_path

  source = ET.SubElement(root, "source")
  ET.SubElement(source, "database").text = "Unknown"

  size = ET.SubElement(root, "size")
  ET.SubElement(size, "width").text = str(width)
  ET.SubElement(size, "height").text = str(height)
  ET.SubElement(size, "depth").text = str(depth)

  ET.SubElement(root, "segmented").text = "0"

  object = ET.SubElement(root, "object")
  ET.SubElement(object, "name").text = class_name
  ET.SubElement(object, "pose").text = "Unspecified"
  ET.SubElement(object, "truncated").text = "0"
  ET.SubElement(object, "difficult").text = "0"

  bndbox = ET.SubElement(object, "bndbox")
  ET.SubElement(bndbox, "xmin").text = str(bbox[0])
  ET.SubElement(bndbox, "ymin").text = str(bbox[1])
  ET.SubElement(bndbox, "xmax").text = str(bbox[0]+bbox[2])
  ET.SubElement(bndbox, "ymax").text = str(bbox[1]+bbox[3])

  base_name = os.path.splitext(filename)[0]
  base_name = "-".join(os.path.split(base_name))
  anns_filepath = os.path.join(output_folder, base_name + ".xml")

  with open(anns_filepath, "w") as text_file:
    text_file.write(prettify(root))


def generate(args):
  # list annotations from base_folder
  all_anns = common.list_files(args.base_folder, args.limit)

  # set output folder
  output_folder = os.path.join(args.output_folder,
                               "voc") if args.output_folder else "voc"

  # set train folder or test folder
  output_folder = os.path.join(output_folder, "test" if args.test else "train")

  print("Output folder: ", output_folder)

  # check if output folder exists
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for a in all_anns:
    savexml(a, args.base_folder, output_folder,
            ["ssiv_bahia", "jequitaia", "balsa"])
