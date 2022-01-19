import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import copy

class MultipleNegativesRankingLoss(nn.Module):
    """
        1. util.cos_sim已经包含L2_Normlazation
        2. 相似度不使用Einstein sum, 而是使用余弦相似度
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, loss_weight: float = 0.15): 
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight          #2021.12.03 联合训练的weight=0.15 , 超参数设置From:https://arxiv.org/abs/2105.11741


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        ### 1. unsup-simcse, in-batch negative samples
        word_emb_model = self.model._first_module()
        for sentence_feature in sentence_features:
            word_emb_model(sentence_feature)
        pooling_model = self.model._last_module()
        sentence_emb = [pooling_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        logits_simcse = self.similarity_fct(sentence_emb[0], sentence_emb[1])  # [bs,bs]

        ### 2. cal sim, the negative samples from moco-queue
        encoder_k_model = self.model[2]
        q_embs = sentence_emb[0]
        queue_embs = encoder_k_model.queue.clone().detach()
        logits_neg = self.similarity_fct(q_embs, queue_embs)
        
        ### 3. cross-entropy with in-batch and history-batch
        labels = torch.tensor(range(len(logits_simcse)), dtype=torch.long, device=logits_simcse.device)   # 正例为[bs,bs]对角线; 其余scores均为负例
        scores = torch.cat([logits_simcse, logits_neg], dim=1)  * self.scale
        
        ### 4. update moco-queue
        # features_copy = copy.deepcopy(sentence_features[0].detach())
        k_embs = encoder_k_model(sentence_features[0])['k_sentence_embeddings']
        encoder_k_model._dequeue_and_enqueue(k_embs)

        return self.cross_entropy_loss(scores, labels) * self.loss_weight

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__, 'loss_weight': self.loss_weight}





