"""Native mask supervision evaluated only inside each assigned box.

Bilinear interpolation is linear in the mask coefficients: interpolate the
single low-resolution logit, instead of multiplying 32 native-sized prototypes
for every positive anchor. Border padding matches interpolate(align_corners=False).
FP32 arithmetic is explicit; parity is against the FP32 objective.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from shared.chunked_mask_loss import ChunkedSegmentationLoss
from ultralytics.utils.ops import xyxy2xywh


class ROISegmentationLoss(ChunkedSegmentationLoss):
    def loss(self,preds,batch):
        proto=preds['proto']
        if not isinstance(proto,torch.Tensor) or proto.ndim!=4:
            raise ValueError('YOLO11 tensor prototypes required')
        coefficients=preds['mask_coefficient'].permute(0,2,1).contiguous()
        (fg,indices,boxes,_,_),detection,_=self.get_assigned_targets_and_loss(preds,batch)
        losses=torch.zeros(5,device=self.device)
        losses[0],losses[2],losses[3]=detection[0],detection[1],detection[2]
        masks=batch['masks'].to(self.device).float()
        size=torch.tensor(preds['feats'][0].shape[2:],device=self.device,dtype=torch.float32)*self.stride[0]
        losses[1]=self.calculate_segmentation_loss(fg,masks,indices,boxes,
            batch['batch_idx'].view(-1,1),proto,coefficients,size)*self.hyp.box
        return losses*proto.shape[0],losses.detach()

    def _roi(self,coefficient,proto,masks,target,box,area,height,width):
        x0,y0,x1,y1=box
        with torch.autocast(device_type=proto.device.type,enabled=False):
            low=(coefficient.float()@proto.float().flatten(1)).reshape(1,1,*proto.shape[-2:])
            x=(torch.arange(x0,x1,device=proto.device,dtype=torch.float32)+.5)*(2/width)-1
            y=(torch.arange(y0,y1,device=proto.device,dtype=torch.float32)+.5)*(2/height)-1
            yy,xx=torch.meshgrid(y,x,indexing='ij')
            grid=torch.stack([xx,yy],dim=-1)[None]
            logits=F.grid_sample(low,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0]
            truth=((masks[y0:y1,x0:x1]==target+1).float() if self.overlap
                   else masks[target,y0:y1,x0:x1].float())
            return F.binary_cross_entropy_with_logits(logits,truth,reduction='sum')/(height*width*area.float())

    def calculate_segmentation_loss(self,fg_mask,masks,target_gt_idx,target_bboxes,
                                    batch_idx,proto,pred_masks,imgsz):
        height,width=masks.shape[-2:]
        normalized=target_bboxes.float()/imgsz[[1,0,1,0]].float()
        areas=xyxy2xywh(normalized)[...,2:].prod(2)
        boxes=normalized*normalized.new_tensor([width,height,width,height])
        loss=proto.reshape(-1)[0]*0+pred_masks.reshape(-1)[0]*0
        for i in range(len(fg_mask)):
            positive=fg_mask[i].nonzero(as_tuple=True)[0]
            # One CPU transfer per image, not per anchor. Crop bounds never have
            # gradients in the source objective (assignment is already detached).
            rounded=(boxes[i,positive].ceil() if proto.is_cuda or len(positive)>=50
                     else boxes[i,positive].round())
            rounded=rounded.clamp_min(0).minimum(rounded.new_tensor([width,height,width,height]))
            bounds=rounded.to(torch.int64).tolist()
            targets=target_gt_idx[i,positive].tolist()
            anchors=positive.tolist()
            image_masks=masks[i] if self.overlap else masks[batch_idx.view(-1)==i]
            for anchor,target,box in zip(anchors,targets,bounds):
                x0,y0,x1,y1=box
                if x1<=x0 or y1<=y0:
                    continue
                inputs=(pred_masks[i,anchor],proto[i],image_masks,target,box,areas[i,anchor],height,width)
                loss=loss+(checkpoint(self._roi,*inputs,use_reentrant=False)
                           if torch.is_grad_enabled() else self._roi(*inputs))
        return loss/fg_mask.sum().clamp_min(1)
