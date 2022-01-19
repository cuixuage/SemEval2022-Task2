import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class MultipleNegativesRankingLoss(nn.Module):
    """
        REF: simcse
        https://colab.research.google.com/drive/1gAjXcI4uSxDE_IcvZdswFYVAo7XvPeoU?usp=sharing#scrollTo=UXUsikOc6oiB
    """
    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, loss_weight: float = 0.15): 
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLoss, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.loss_weight = loss_weight          #2021.12.03 联合训练的weight=0.15 , 超参数设置From:https://arxiv.org/abs/2105.11741


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        # 1.获取token_emb
        word_emb_model = self.model._first_module()
        for sentence_feature in sentence_features:
            word_emb_model(sentence_feature)
        # 2. 通过pooling, 再计算相似度
        pooling_model = self.model._last_module()
        sentence_emb = [pooling_model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]

        scores = self.similarity_fct(sentence_emb[0], sentence_emb[1]) * self.scale       ### 2021.12.1 scores = [bs, bs] simcse论文,温度系数为 0.05, 也就是乘以20
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  ### 2021.12.1  len(tenor) = print the size of the first dimension = bs; simcse论文,自监督, label=[bs,1],values=[0,1,2,3,4,5..]对角线
        return self.cross_entropy_loss(scores, labels) * self.loss_weight

    def get_config_dict(self):
        return {'scale': self.scale, 'similarity_fct': self.similarity_fct.__name__, 'loss_weight': self.loss_weight}





