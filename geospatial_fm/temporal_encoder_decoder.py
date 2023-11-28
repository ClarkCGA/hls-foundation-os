# Copyright (c) OpenMMLab. All rights reserved.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from mmseg.core import add_prefix
# from mmseg.ops import resize
# from mmseg.models import builder
# from mmseg.models.builder import SEGMENTORS
# from mmseg.models.segmentors.base import BaseSegmentor
# from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


# @SEGMENTORS.register_module()
# class TemporalEncoderDecoder(EncoderDecoder):
#     """Encoder Decoder segmentors.

#     EncoderDecoder typically consists of backbone, neck, decode_head, auxiliary_head.
#     Note that auxiliary_head is only used for deep supervision during training,
#     which could be dumped during inference.

#     The backbone should return plain embeddings.
#     The neck can process these to make them suitable for the chosen heads.
#     The heads perform the final processing that will return the output.
#     """

#     def __init__(self,
#                  backbone,
#                  decode_head,
#                  neck=None,
#                  auxiliary_head=None,
#                  train_cfg=None,
#                  test_cfg=None,
#                  pretrained=None,
#                  init_cfg=None,
#                  frozen_backbone=False):
#         super(EncoderDecoder, self).__init__(init_cfg)
#         if pretrained is not None:
#             assert backbone.get('pretrained') is None, \
#                 'both backbone and segmentor set pretrained weight'
#             backbone.pretrained = pretrained
#         self.backbone = builder.build_backbone(backbone)
        
#         if frozen_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         if neck is not None:
#             self.neck = builder.build_neck(neck)
#         self._init_decode_head(decode_head)
#         self._init_auxiliary_head(auxiliary_head)

#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg
#         assert self.with_decode_head

#     def encode_decode(self, img, img_metas):
#         """Encode images with backbone and decode into a semantic segmentation
#         map of the same size as input."""
#         x = self.extract_feat(img)
#         out = self._decode_head_forward_test(x, img_metas)
        
#         #### size calculated over last two dimensions ###
#         size = img.shape[-2:]
        
#         out = resize(
#             input=out,
#             size=size,
#             mode='bilinear',
#             align_corners=self.align_corners)
#         return out
      
#     def slide_inference(self, img, img_meta, rescale):
#         """Inference by sliding-window with overlap.

#         If h_crop > h_img or w_crop > w_img, the small patch will be used to
#         decode without padding.
#         """

#         h_stride, w_stride = self.test_cfg.stride
#         h_crop, w_crop = self.test_cfg.crop_size
        
#         #### size and bactch size over last two dimensions ###
#         img_size = img.size()
#         batch_size = img_size[0]
#         h_img = img_size[-2]
#         w_img = img_size[-1]
#         out_channels = self.out_channels
#         h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
#         w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
#         preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
#         count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
#         for h_idx in range(h_grids):
#             for w_idx in range(w_grids):
#                 y1 = h_idx * h_stride
#                 x1 = w_idx * w_stride
#                 y2 = min(y1 + h_crop, h_img)
#                 x2 = min(x1 + w_crop, w_img)
#                 y1 = max(y2 - h_crop, 0)
#                 x1 = max(x2 - w_crop, 0)
                
#                 if len(img_size) == 4:
                    
#                     crop_img = img[:, :, y1:y2, x1:x2]
                
#                 elif len(img_size) == 5:
                    
#                     crop_img = img[:, :, :, y1:y2, x1:x2]
                
                
                
#                 crop_seg_logit = self.encode_decode(crop_img, img_meta)
#                 preds += F.pad(crop_seg_logit,
#                                (int(x1), int(preds.shape[3] - x2), int(y1),
#                                 int(preds.shape[2] - y2)))

#                 count_mat[:, :, y1:y2, x1:x2] += 1
#         assert (count_mat == 0).sum() == 0
#         if torch.onnx.is_in_onnx_export():
#             # cast count_mat to constant while exporting to ONNX
#             count_mat = torch.from_numpy(
#                 count_mat.cpu().detach().numpy()).to(device=img.device)
#         preds = preds / count_mat

#         if rescale:
#             # remove padding area
#             #### size over last two dimensions ###
#             resize_shape = img_meta[0]['img_shape'][:2]
#             preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
#             preds = resize(
#                 preds,
#                 size=img_meta[0]['ori_shape'][:2],
#                 mode='bilinear',
#                 align_corners=self.align_corners,
#                 warning=False)
#         return preds

#     def whole_inference(self, img, img_meta, rescale):
#         """Inference with full image."""

#         seg_logit = self.encode_decode(img, img_meta)
#         if rescale:
#             # support dynamic shape for onnx
#             if torch.onnx.is_in_onnx_export():
#                 size = img.shape[-2:]
#             else:
#                 # remove padding area
#                 resize_shape = img_meta[0]['img_shape'][:2] 
#                 seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
#                 size = img_meta[0]['ori_shape'][:2]
#             seg_logit = resize(
#                 seg_logit,
#                 size=size,
#                 mode='bilinear',
#                 align_corners=self.align_corners,
#                 warning=False)

#         return seg_logit

#     def inference(self, img, img_meta, rescale):
#         """Inference with slide/whole style.

#         Args:
#             img (Tensor): The input image of shape (N, 3, H, W).
#             img_meta (dict): Image info dict where each dict has: 'img_shape',
#                 'scale_factor', 'flip', and may also contain
#                 'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
#                 For details on the values of these keys see
#                 `mmseg/datasets/pipelines/formatting.py:Collect`.
#             rescale (bool): Whether rescale back to original shape.

#         Returns:
#             Tensor: The output segmentation map.
#         """

#         assert self.test_cfg.mode in ['slide', 'whole']
#         ori_shape = img_meta[0]['ori_shape']
#         assert all(_['ori_shape'] == ori_shape for _ in img_meta)
#         if self.test_cfg.mode == 'slide':
#             seg_logit = self.slide_inference(img, img_meta, rescale)
#         else:
#             seg_logit = self.whole_inference(img, img_meta, rescale)
            
#         if self.out_channels == 1:
#             output = F.sigmoid(seg_logit)
#         else:
#             output = F.softmax(seg_logit, dim=1)

#         flip = (
#             img_meta[0]["flip"] if "flip" in img_meta[0] else False
#         )  ##### if flip key is not there d not apply it
#         if flip:
#             flip_direction = img_meta[0]["flip_direction"]
#             assert flip_direction in ["horizontal", "vertical"]
#             if flip_direction == "horizontal":
#                 output = output.flip(dims=(3,))
#             elif flip_direction == "vertical":
#                 output = output.flip(dims=(2,))
#         return output

#     def simple_test(self, img, img_meta, rescale=True):
#         """Simple test with single image."""
#         seg_logit = self.inference(img, img_meta, rescale)
#         if self.out_channels == 1:
#             seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
#         else:
#             seg_pred = seg_logit.argmax(dim=1)
#         if torch.onnx.is_in_onnx_export():

#             seg_pred = seg_pred.unsqueeze(0)
#             return seg_pred
#         seg_pred = seg_pred.cpu().numpy()
#         # unravel batch dim
#         seg_pred = list(seg_pred)
#         return seg_pred

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from mmpretrain.registry import MODELS
# from mmpretrain.models import builder
# from mmengine.model import BaseModel


# @MODELS.register_module()
# class GeospatialMultiLabelClassifier(BaseModel):
#     """Classifier based on TemporalViTEncoder for multi-label classification tasks."""

#     def __init__(self,
#                  backbone,
#                  cls_head,
#                  pretrained=None,
#                  frozen_backbone=False):
#         super(GeospatialMultiLabelClassifier, self).__init__()

#         if pretrained is not None:
#             backbone.pretrained = pretrained
#         self.backbone = builder.build_backbone(backbone)
        
#         if frozen_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False

#         # Add the MultiLabelClsHead
#         self.head = builder.build_head(cls_head)


#     def forward(self, img, lbl=None, mode='loss'):

#         features = self.backbone(img)
#         cls_embeddings = features[0][:, 0, :]
        
#         if mode == 'loss':
#             output = self.head(cls_embeddings, lbl)
#             return {'loss': output}
#         elif mode in ['predict', 'tensor']:
#             output = self.cls_head(cls_embeddings)
#             return output



from mmpretrain.models.builder import BACKBONES, HEADS
from mmpretrain.models.heads import MultiLabelClsHead
from mmengine.registry import MODELS
from mmpretrain.models import builder
from mmengine.model import BaseModel

from mmengine import Registry

#HEADS = Registry('head', scope='mmengine', locations=['mmengine.models.heads'])
#MODELS.register_module()(MultiLabelClsHead)


# from mmpretrain.models.losses.cross_entropy_loss import  cross_entropy, soft_cross_entropy, binary_cross_entropy
# from mmpretrain.models.losses.utils import weight_reduce_loss
# import torch.nn as nn
# import torch.nn.functional as F

# from mmpretrain.models.losses.cross_entropy_loss import CrossEntropyLoss
# LOSSES = Registry('loss', scope='mmengine', locations=['mmengine.models.losses'])
# LOSSES.register_module()(CrossEntropyLoss)



# @LOSSES.register_module()
# class CrossEntropyLoss(nn.Module):
#     """Cross entropy loss.

#     Args:
#         use_sigmoid (bool): Whether the prediction uses sigmoid
#             of softmax. Defaults to False.
#         use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
#             Defaults to False.
#         reduction (str): The method used to reduce the loss.
#             Options are "none", "mean" and "sum". Defaults to 'mean'.
#         loss_weight (float):  Weight of the loss. Defaults to 1.0.
#         class_weight (List[float], optional): The weight for each class with
#             shape (C), C is the number of classes. Default None.
#         pos_weight (List[float], optional): The positive weight for each
#             class with shape (C), C is the number of classes. Only enabled in
#             BCE loss when ``use_sigmoid`` is True. Default None.
#     """

#     def __init__(self,
#                  use_sigmoid=False,
#                  use_soft=False,
#                  reduction='mean',
#                  loss_weight=1.0,
#                  class_weight=None,
#                  pos_weight=None):
#         super(CrossEntropyLoss, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         self.use_soft = use_soft
#         assert not (
#             self.use_soft and self.use_sigmoid
#         ), 'use_sigmoid and use_soft could not be set simultaneously'

#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.class_weight = class_weight
#         self.pos_weight = pos_weight

#         if self.use_sigmoid:
#             self.cls_criterion = binary_cross_entropy
#         elif self.use_soft:
#             self.cls_criterion = soft_cross_entropy
#         else:
#             self.cls_criterion = cross_entropy

#     def forward(self,
#                 cls_score,
#                 label,
#                 weight=None,
#                 avg_factor=None,
#                 reduction_override=None,
#                 **kwargs):
#         assert reduction_override in (None, 'none', 'mean', 'sum')
#         reduction = (
#             reduction_override if reduction_override else self.reduction)

#         if self.class_weight is not None:
#             class_weight = cls_score.new_tensor(self.class_weight)
#         else:
#             class_weight = None

#         # only BCE loss has pos_weight
#         if self.pos_weight is not None and self.use_sigmoid:
#             pos_weight = cls_score.new_tensor(self.pos_weight)
#             kwargs.update({'pos_weight': pos_weight})
#         else:
#             pos_weight = None

#         loss_cls = self.loss_weight * self.cls_criterion(
#             cls_score,
#             label,
#             weight,
#             class_weight=class_weight,
#             reduction=reduction,
#             avg_factor=avg_factor,
#             **kwargs)
#         return loss_cls



@MODELS.register_module()
class GeospatialMultiLabelClassifier(BaseModel):
    """Classifier based on TemporalViTEncoder for multi-label classification tasks."""

    def __init__(self,
                 backbone,
                 cls_head,
                 pretrained=None,
                 frozen_backbone=False):
        super(GeospatialMultiLabelClassifier, self).__init__()

        if pretrained is not None:
            backbone.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        
        if frozen_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Add the MultiLabelClsHead
        self.head = builder.build_head(cls_head)


    def forward(self, img, lbl=None, mode='loss'):

        features = self.backbone(img)
        cls_embeddings = features[0][:, 0, :]
        
        if mode == 'loss':
            output = self.head(cls_embeddings, lbl)
            return {'loss': output}
        elif mode in ['predict', 'tensor']:
            output = self.cls_head(cls_embeddings)
            return output
