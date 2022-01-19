import torch
import math
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class CosineSimilarityCrossAttentionLoss(nn.Module):
    """
    """
    def __init__(self, model: SentenceTransformer, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityCrossAttentionLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def update_features(self, sen_a_emb, sen_b_emb, tokens_masks):
        samples = []
        simple1 = dict()
        simple1['token_embeddings'] = sen_a_emb
        simple1['cls_token_embeddings'] = sen_a_emb[:0:]
        simple1['attention_mask'] = tokens_masks[0]
        samples.append(simple1)
        simple2 = dict()
        simple2['token_embeddings'] = sen_b_emb
        simple2['cls_token_embeddings'] = sen_b_emb[:0:]
        simple2['attention_mask'] = tokens_masks[1]
        samples.append(simple2)
        return samples

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        """
            Cross Attention思路来自于 https://github.com/Decem-Y/sohu_text_matching_Rank2/  
            (但是其实现Mask矩阵的操作很有可能是错误的)
        """
        # 1.获取token_emb
        word_emb_model = self.model._first_module()
        embeddings = [word_emb_model(sentence_feature)['token_embeddings'] for sentence_feature in sentence_features ] # 2021.12.27 [bs, seq_len_max_in_batch, embed_dim]  32,35,768
        tokens_masks = [ sentence_feature['attention_mask'] for sentence_feature in sentence_features ] # 2021.12.27 [bs, seq_len_max_in_batch]  32,35
        
        # 2.计算cross_attention
        cross_model = self.model[1]
        extended_attention_masks = []
        for item in tokens_masks:
            attention_mask = item.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            extended_attention_masks.append(attention_mask)
        sen_a_emb = cross_model(features=sentence_features[0], hidden_states=embeddings[0],
                                encoder_hidden_states=embeddings[1], encoder_attention_mask=extended_attention_masks[1])['cross_token_embeddings']
        sen_b_emb = cross_model(features=sentence_features[1], hidden_states=embeddings[1],
                                encoder_hidden_states=embeddings[0], encoder_attention_mask=extended_attention_masks[0])['cross_token_embeddings']
        # print(tokens_masks[0].size(), tokens_masks[1].size(), extended_attention_masks[0].size(), extended_attention_masks[1].size()) # torch.Size([32, 41]) torch.Size([32, 39]) torch.Size([32, 1, 1, 41]) torch.Size([32, 1, 1, 39])
        # print(embeddings[0].size(), embeddings[1].size()) # torch.Size([32, 41, 768]) torch.Size([32, 39, 768])
        # print(sen_a_emb.size(), sen_b_emb.size()) # torch.Size([32, 41, 768]) torch.Size([32, 39, 768])

        # 3. 通过pooling, 再计算相似度
        pooling_model = self.model._last_module()
        samples = self.update_features(sen_a_emb, sen_b_emb, tokens_masks)
        # print(samples[0]['token_embeddings'].size(), samples[0]['attention_mask'].size(), samples[1]['token_embeddings'].size(), samples[1]['attention_mask'].size())  #torch.Size([32, 41, 768]) torch.Size([32, 41]) torch.Size([32, 39, 768]) torch.Size([32, 39])
        sentence_emb = [pooling_model(sample)['sentence_embedding'] for sample in samples]

        output = self.cos_score_transformation(torch.cosine_similarity(sentence_emb[0], sentence_emb[1]))
        return self.loss_fct(output, labels.view(-1))

