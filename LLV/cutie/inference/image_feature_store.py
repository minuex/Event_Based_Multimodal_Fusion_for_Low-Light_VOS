import warnings
from typing import Iterable
import torch
from cutie.model.cutie import CUTIE
import logging

log = logging.getLogger()

class ImageFeatureStore:
    """
    A cache for image features.
    These features might be reused at different parts of the inference pipeline.
    This class provide an interface for reusing these features.
    It is the user's responsibility to delete redundant features.

    Feature of a frame should be associated with a unique index -- typically the frame id.
    """
    def __init__(self, network: CUTIE, no_warning: bool = False):
        self.network = network
        self._store = {}
        self.no_warning = no_warning

    def _encode_feature(self, index: int, image: torch.Tensor) -> None:
        ms_features, pix_feat = self.network.encode_image(image)
        key, shrinkage, selection = self.network.transform_key(ms_features[0])
        self._store[index] = (ms_features, pix_feat, key, shrinkage, selection)

    def get_features(self, index: int,
                     image: torch.Tensor) -> (Iterable[torch.Tensor], torch.Tensor):
        if index not in self._store:
            self._encode_feature(index, image)

        return self._store[index][:2]

    def get_key(self, index: int,
                image: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if index not in self._store:
            self._encode_feature(index, image)

        return self._store[index][2:]

    def delete(self, index: int) -> None:
        if index in self._store:
            del self._store[index]

    def __len__(self):
        return len(self._store)

    def __del__(self):
        if len(self._store) > 0 and not self.no_warning:
            warnings.warn(f'Leaking {self._store.keys()} in the image feature store')

    def encode_image_with_event(self, image: torch.Tensor, event: torch.Tensor):
        """
        입력 이미지를 pixel encoder로 인코딩하고,
        event feature를 event encoder로 인코딩한 뒤,
        ACMF로 융합하여 ms_feature, pixel feature를 반환한다.
        """
        # 1. 인코딩
        ms_feat, pix_feat = self.network.encode_image(image)       # RGB로부터
        evt_feat = self.network.encode_event(event)              # Event로부터
        log.info(f"[evt_feat] shape: {evt_feat.shape}")

        # 2. 융합
        fused_feat = self.network.forward_acmf(pix_feat, evt_feat)

        # 3. (기존 pipeline에서 필요하던 output만 반환)
        return ms_feat, fused_feat

    def _encode_feature_plus_event(self, index: int, image: torch.Tensor, event: torch.Tensor):
        """
        RGB + Event 데이터를 기반으로 특징을 인코딩하고 저장소에 캐시한 뒤,
        기존 get_features와 동일한 구조로 반환한다.
        """
        ms_feat, fused_feat = self.encode_image_with_event(image, event)
        key, shrinkage, selection = self.network.transform_key(ms_feat[0])
        self._store[index] = (ms_feat, fused_feat, key, shrinkage, selection)
        return ms_feat, fused_feat
    
    def get_features_plus_event(self, index: int, image: torch.Tensor, event: torch.Tensor):
        if index not in self._store:
            return self._encode_feature_plus_event(self, index, image, event)
        return self._store[index][:2]
