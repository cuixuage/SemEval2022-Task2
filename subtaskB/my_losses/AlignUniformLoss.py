import torch
from torch import nn, Tensor
from typing import Iterable, Dict
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class AlignUniformLoss(nn.Module):
    """
    REF:
        https://github.com/SsnL/align_uniform
        https://github.com/princeton-nlp/SimCSE/issues/85
        https://github.com/princeton-nlp/SimCSE/issues/41
    """
    def __init__(self, model: SentenceTransformer,  alpha=2, t=2, lamda=0.5, loss_weight=0.15): 
        """
        :param model: SentenceTransformer model
        :param alpha: 
        :param alpha: 
        """
        super(AlignUniformLoss, self).__init__()
        self.model = model
        self.lamda = lamda
        self.alpha = alpha
        self.t = t
        self.loss_weight = loss_weight

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        def align_loss(x, y, alpha=2):    
            return (x - y).norm(p=2, dim=1).pow(alpha).mean()
        def uniform_loss(x, t=2):
            return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        z1 = F.normalize(embeddings_a, p=2, dim=1)
        z2 = F.normalize(embeddings_b, p=2, dim=1)
        z = torch.cat((z1,z2), dim=-1)
        a_loss = align_loss(z1, z2, alpha=self.alpha)
        u_loss = uniform_loss(z, t=self.t)
        return (self.lamda * a_loss + (1 - self.lamda) * u_loss) * self.loss_weight

    def get_config_dict(self):
        return {'lamda': self.lamda, 'alpha': self.alpha, 't': self.t, 'loss_weight': self.loss_weight}




