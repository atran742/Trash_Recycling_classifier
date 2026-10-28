"""
recording_names.py - the file names of capture_photos.py video mode, in one place.

A recording saves one frame every 2 seconds, named <class>_item<NN>_f<frame>.jpg, for
example trash_item03_f2.jpg: item 3, frame 2. capture_photos.py uses this to pick the next
item number, and compare_models.py uses it to score the frames of one recording as one item.
It is a tiny file of its own so that both can import it without loading the webcam code.
"""

import re

# (?P<name>...) gives a part of the match a name, so found["item"] reads the item number.
# re.IGNORECASE: a frame renamed to TRASH_ITEM03_F2.JPG still counts as a recording frame.
ITEM_NAME = re.compile(r"(?P<label>[a-z]+)_item(?P<item>\d+)_f(?P<frame>\d+)\.jpe?g", re.IGNORECASE)
