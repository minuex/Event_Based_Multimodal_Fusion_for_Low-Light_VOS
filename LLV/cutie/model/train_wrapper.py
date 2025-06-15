import logging
from omegaconf import DictConfig
import numpy as np
import torch
from typing import Dict

from einops.layers.torch import Rearrange
from cutie.model.cutie import CUTIE

#ACMF 삽입
from cutie.model.big_modules import ACMFModule
#ObjectEventFusion 삽입
from cutie.model.big_modules import ObjectEventFusion

log = logging.getLogger()


class CutieTrainWrapper(CUTIE):
    def __init__(self, cfg: DictConfig, stage_cfg: DictConfig):
        super().__init__(cfg, single_object=(stage_cfg.num_objects == 1))

        self.sensory_dim = cfg.model.sensory_dim
        self.seq_length = stage_cfg.seq_length
        self.num_ref_frames = stage_cfg.num_ref_frames
        self.deep_update_prob = stage_cfg.deep_update_prob
        self.use_amp = stage_cfg.amp
        self.move_t_out_of_batch = Rearrange('(b t) c h w -> b t c h w', t=self.seq_length)
        self.move_t_from_batch_to_volume = Rearrange('(b t) c h w -> b c t h w', t=self.seq_length)
        #acmf 정의
        self.acmf = ACMFModule(feature_dim=cfg.model.pixel_dim)
        #object + event 융합 정의
        self.obj_evt_fuser = ObjectEventFusion(embed_dim=cfg.model.value_dim) 

    def forward(self, data: Dict):
        out = {}
        frames = data['rgb']
        event_voxel = data['event']  # ← 추가됨
        first_frame_gt = data['first_frame_gt'].float()
        b, seq_length = frames.shape[:2]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        max_num_objects = max(num_filled_objects)
        first_frame_gt = first_frame_gt[:, :, :max_num_objects]
        selector = data['selector'][:, :max_num_objects].unsqueeze(2).unsqueeze(2)

        num_objects = first_frame_gt.shape[2]
        out['num_filled_objects'] = num_filled_objects

        def get_ms_feat_ti(ti):
            return [f[:, ti] for f in ms_feat]

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            frames_flat = frames.view(b * seq_length, *frames.shape[2:])
            ms_feat, pix_feat = self.encode_image(frames_flat)

            # Event feature 처리
            event_feat_raw = self.event_encoder(event_voxel.view(b * seq_length, 5, *frames.shape[-2:]))
            event_feat = self.event_feat_proj(event_feat_raw)
            event_feat = self.move_t_out_of_batch(event_feat)  # shape: [B * T, C, H, W]

            with torch.cuda.amp.autocast(enabled=False):
                keys, shrinkages, selections = self.transform_key(ms_feat[0].float())

            h, w = keys.shape[-2:]
            keys = self.move_t_from_batch_to_volume(keys)
            shrinkages = self.move_t_from_batch_to_volume(shrinkages)
            selections = self.move_t_from_batch_to_volume(selections)
            ms_feat = [self.move_t_out_of_batch(f) for f in ms_feat]
            pix_feat = self.move_t_out_of_batch(pix_feat)

            # zero-init sensory
            sensory = torch.zeros((b, num_objects, self.sensory_dim, h, w), device=frames.device)
            fused_feat_0 = self.acmf(pix_feat[:, 0], event_feat[:, 0])
            msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, 0], fused_feat_0, sensory,
                                                            first_frame_gt[:, 0])
            masks = first_frame_gt[:, 0]

            msk_values = msk_val.unsqueeze(3)
            #obj_values = obj_val.unsqueeze(2) if obj_val is not None else None
            #
            #첫 프레임 융합=======================================================
            if obj_val is not None:
                # event_feat[:, 0] → [B, C, H, W]
                obj_val = self.obj_evt_fuser(obj_val, event_feat[:, 0])
                obj_values = obj_val.unsqueeze(2)
            else:
                obj_values = None
            #===================================================================
            for ti in range(1, seq_length):
                if ti <= self.num_ref_frames:
                    ref_msk_values = msk_values
                    ref_keys = keys[:, :, :ti]
                    ref_shrinkages = shrinkages[:, :, :ti] if shrinkages is not None else None
                else:
                    ridx = [torch.randperm(ti)[:self.num_ref_frames] for _ in range(b)]
                    ref_msk_values = torch.stack(
                        [msk_values[bi, :, :, ridx[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([keys[bi, :, ridx[bi]] for bi in range(b)], 0)
                    ref_shrinkages = torch.stack([shrinkages[bi, :, ridx[bi]] for bi in range(b)], 0)

                fused_feat_ti = self.acmf(pix_feat[:, ti], event_feat[:, ti])

                readout, aux_input = self.read_memory(keys[:, :, ti], selections[:, :, ti],
                                                      ref_keys, ref_shrinkages, ref_msk_values,
                                                      obj_values, fused_feat_ti,
                                                      sensory, masks, selector)

                aux_output = self.compute_aux(fused_feat_ti, aux_input, selector)
                sensory, logits, masks = self.segment(get_ms_feat_ti(ti), readout, sensory, selector=selector)
                masks = masks[:, 1:]

                if ti < (self.seq_length - 1):
                    deep_update = np.random.rand() < self.deep_update_prob
                    fused_feat_ti = self.acmf(pix_feat[:, ti], event_feat[:, ti])
                    msk_val, sensory, obj_val, _ = self.encode_mask(frames[:, ti], fused_feat_ti,
                                                                    sensory, masks,
                                                                    deep_update=deep_update)
                    msk_values = torch.cat([msk_values, msk_val.unsqueeze(3)], 3)
                    #if obj_val is not None:
                    #    obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)], 2)
                    #이후 프레임 융합 추가=================================================
                    if obj_val is not None:
                        obj_val = self.obj_evt_fuser(obj_val, event_feat[:, ti])
                        obj_values = torch.cat([obj_values, obj_val.unsqueeze(2)], 2)
                    #===================================================================

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits
                out[f'aux_{ti}'] = aux_output

        return out