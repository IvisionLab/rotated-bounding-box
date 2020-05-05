# Rotated object detection

This code uses Mask-RCNN implementation of https://github.com/matterport/Mask_RCNN as base.

### The source code will be available soon.

RBoxNet is a CNN to detect rotated bounding box. Your first goal was to detect
objects in images of forward-looking sonars. RBoxNet modifies the removes the semantic
segmentation part from Mask-RCNN and include news layers for rotated bounding box.

We have tested here three different approaches for rotated bounding box detections.

* Vertices (verts)
* Box offsets (deltas)
* Orientation and Size (rotdim)