import torch
from torch import Tensor
from torch import nn
from sentence_transformers import models
from typing import Union, Tuple, List, Iterable, Dict
import math
import json
import os

class MoCoBuilder(nn.Module):
    """
    1. Moco
        https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    2. InfoXLM
        https://github.com/microsoft/unilm/blob/master/infoxlm/src-infoxlm/infoxlm/criterions/xlco.py
    """
    def __init__(self, encoder_q:models.Transformer, encoder_k:models.Transformer,dim=768, T=0.05, K=int(32*2.5), M=0.999):
        super(MoCoBuilder, self).__init__()
        """
            dim: feature dimension (default: 768)
            K: queue size; number of negative keys (default: 2.5*batchsize, From ESimCSE)
            m: moco momentum of updating key encoder (default: 0.999)
            T: softmax temperature (default: 0.05)
        """
        self.queue_size = K
        self.M = M
        self.T = T
        self.dim = dim
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(self.queue_size, self.dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.M + param_q.data * (1. - self.M)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
            https://github.com/microsoft/unilm/blob/master/infoxlm/src-infoxlm/infoxlm/models/infoxlm.py#L78
        """
        batch_size = keys.size(0)
        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0
        if ptr + batch_size <= self.queue_size:
            self.queue[ptr:ptr+batch_size, :] = keys
            ptr = (ptr + batch_size) % self.queue_size
        else:
            left_len = self.queue_size - ptr
            self.queue[ptr:, :] = keys[:left_len, :]
            ptr = batch_size-left_len
            self.queue[:ptr, :] = keys[left_len:, :]
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def forward(
        self,
        features: Dict[str, Tensor]
    ):
        """
            1. 使用Encoder-K编码语句的sentence-embedding
            2. 更新Encoder-K权重参数
        """
        with torch.no_grad():
            output_features = self.encoder_k(features)
            token_embeddings = output_features['token_embeddings']
            attention_mask = output_features['attention_mask']

            output_vectors = []
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
            sentence_embedding = torch.cat(output_vectors, 1)

            self._momentum_update_key_encoder()  # update the key encoder
            features.update({'k_sentence_embeddings': sentence_embedding})
        return features

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'moco_builder_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)
        ## 2022.1.10 无需保存权重
        # torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))
    
    def get_config_dict(self):
        return {'dim': self.dim, 'T': self.T, 'K': self.queue_size, 'M': self.M}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'moco_builder_config.json'), 'r') as fIn:
            config = json.load(fIn)
        # weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        # model = MoCoBuilder(**config)
        # model.load_state_dict(weights)
        # return model
        return


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output