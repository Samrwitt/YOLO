"""
Quick map of where our YOLO-ish detector lives (everything is plain PyTorch).

- `yolo_v1_tiny.py` — the actual network: conv stack, then a 1×1 head that outputs an S×S
  grid of (box stuff + objectness + class logits).
- `yolo_loss.py` — turns predictions + VOC targets into a single scalar loss.
- `datasets/voc_yolo.py` — downloads / reads VOC2007 and builds tensors the loss expects.

Training + mAP eval are in `notebooks/yolo_train_eval.ipynb` so you can run end-to-end in one place.

Why people cared about YOLO: older detectors often ran a classifier on thousands of proposed
regions — slow and fiddly. YOLO said "predict boxes and classes everywhere in one shot."

LeNet in `lenet.py` is the opposite problem setup: one image → one label vector. Here it's
one image → a whole field of predictions, which is what you need when multiple objects show up.

We deliberately stay on a **single** grid (old-school YOLOv1 vibe). Real modern YOLOs stack
pyramids, better matching, fancier augmentations, etc. — way more code than we want for a lab.
"""
