from .coco import get_dataset as get_coco_dataset
from .flickr import get_dataset as get_flickr_dataset
from .docornot import get_dataset as get_docornot_dataset

# from .textcaps import get_dataset as get_textcaps_dataset

DATASETS = {
    "flickr": get_flickr_dataset,
    # "coco": get_coco_dataset,
    "docornot": get_docornot_dataset,
    # "textcaps": get_textcaps_dataset,
}
