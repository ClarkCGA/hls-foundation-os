from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from .geospatial_pipelines import LoadGeospatialAnnotations

        
@DATASETS.register_module()
class GeospatialDataset(CustomDataset):
    """GeospatialDataset dataset.
    """

    def __init__(self, CLASSES=(0, 1), PALETTE=None, **kwargs):
        
        self.CLASSES = CLASSES

        self.PALETTE = PALETTE
        
        gt_seg_map_loader_cfg = kwargs.pop('gt_seg_map_loader_cfg') if 'gt_seg_map_loader_cfg' in kwargs else dict()
        reduce_zero_label = kwargs.pop('reduce_zero_label') if 'reduce_zero_label' in kwargs else False
        
        super(GeospatialDataset, self).__init__(
            reduce_zero_label=reduce_zero_label,
            # ignore_index=2,
            **kwargs)

        self.gt_seg_map_loader = LoadGeospatialAnnotations(reduce_zero_label=reduce_zero_label, **gt_seg_map_loader_cfg)


from mmpretrain.registry import DATASETS
from mmpretrain.datasets.base_dataset import BaseDataset
from typing import List, Any

@DATASETS.register_module()
class MultiLabelDataset(BaseDataset):
    """Multi-label Dataset for image classification.

    This dataset extends BaseDataset to support multi-label classification.

    Args:
        same arguments as BaseDataset
    
    The annotation format is shown as follows.

    .. code-block:: none

        {
            "metainfo":
            {
              "classes": ['A', 'B', 'C', ...]
            },
            "data_list":
            [
              {
                "img_path": "test_img1.jpg",
                'gt_label': [0, 1],
              },
              {
                "img_path": "test_img2.jpg",
                'gt_label': [2],
              },
            ],
            ...
        }
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # You can put any custom initialization here

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: Image categories of specified index.

        Raises:
            KeyError: If 'gt_label' does not exist in data_info
        """
        data_info = self.get_data_info(idx)
        if 'gt_label' not in data_info:
            raise KeyError(f"'gt_label' not found in data_info for index {idx}")
        return data_info['gt_label']
