
# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
# from .geospatial_pipelines import LoadGeospatialAnnotations

        
# @DATASETS.register_module()
# class GeospatialDataset(CustomDataset):
#     """GeospatialDataset dataset.
#     """

#     def __init__(self, CLASSES=(0, 1), PALETTE=None, **kwargs):
        
#         self.CLASSES = CLASSES

#         self.PALETTE = PALETTE
        
#         gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
#         reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
#         super(GeospatialDataset, self).__init__(
#             reduce_zero_label=reduce_zero_label,
#             # ignore_index=2,
#             **kwargs)

#         self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)


import os
#from mmpretrain.registry import DATASETS
from mmengine.registry import DATASETS
from mmpretrain.datasets.base_dataset import BaseDataset
from typing import Optional, Any


@DATASETS.register_module()
class MultiLabelGeospatialDataset(BaseDataset):
    """Multi-label Dataset for image classification.

    This dataset extends BaseDataset to support multi-label classification.

    Args:
        ann_file (str): Annotation file path.
        split (str): Path to the text file containing sample indices for the 
            desired split (train/val).
        metainfo (dict, optional): Meta information for dataset, such as class information.
        ... (Other arguments inherited from BaseDataset)
    """

    def __init__(self, 
                 ann_file: str,
                 split: str,
                 metainfo: Optional[dict] = None,
                 **kwargs: Any):
        # Custom checks or operations for ann_file can go here
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file {ann_file} not found.")
        
        if not ann_file.endswith('.json'):
            raise ValueError("Annotation file must be a .json file")
        
        with open(split, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip().isdigit()]
        
        # Call the parent class's init method
        super().__init__(ann_file=ann_file, metainfo=metainfo, indices=indices, **kwargs)

    def get_cat_ids(self, idx: int) -> list[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: Image categories of specified index.

        """
        data_info = self.get_data_info(idx)
        if 'gt_label' not in data_info:
            raise KeyError(f"'gt_label' not found in data_info for index {idx}")
        
        return data_info['gt_label']