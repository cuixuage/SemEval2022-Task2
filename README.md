# SemEval2022-Task2
Adversarial Training and Contrastive Learning for Multiword Representations  
  
**Abstract:**      
In the SubTaskA, we use InfoXLM as text encoder and exponential moving average (EMA) method and the adversarial attack strategy.    
In the SubTaskB, we add an cross-attention module, contrastive objective and employ a momentum contrast.    
Additionally, we use the alignment and uniformity properties to measure the quality of sentence embeddings.    
  
**Code & Ranking**   
```
+-------------------+---------------------------------------+---------------------+---------+
|      SubTask      | file                                  | Module              | Ranking |
+-------------------+---------------------------------------+---------------------+---------+
|     zero-shot     | no_trainer_zero_shot.py               | EMA + Smart         | 6       |
+-------------------+---------------------------------------+---------------------+---------+
|     one-shot      | no_trainer_one_shot.py                | EMA + FreeLB        | 1       |
+-------------------+---------------------------------------+---------------------+---------+
| pretrain/finetune | CosineSimilarityCrossAttentionLoss.py | MSE Loss            |         |
+-------------------+---------------------------------------+---------------------+---------+
| pretrain/finetune | NegativesRankingLoss_MoCo.py          | InfoNCE Loss + MoCo |         |
+-------------------+---------------------------------------+---------------------+---------+
| pretrain/finetune | CrossAttention.py                     | NetWork             |         | 
+-------------------+---------------------------------------+---------------------+---------+
| pretrain/finetune | MoCoBuilder.py                        | NetWork             |         |
+-------------------+---------------------------------------+---------------------+---------+
|     pretrain      | EmbeddingCrossSimEvaluator.py         | Eval                | 3       |
+-------------------+---------------------------------------+---------------------+---------+
|     finetune      | EmbeddingCrossSimEvaluator.py         | Eval                | 2       |
+-------------------+---------------------------------------+---------------------+---------+
```

**Conclusion**  
![](paper/model.jpg){:height="50%" width="50%"}
![](paper/AandU.jpg){:height="50%" width="50%"}  

i. Dev-F1 score, InfoXLM > XLM-R > M-bert.       
ii. Using the EMA-Method and Adversarial-Training can improve the model robustness.    
iii. Contrastive learning can further improve the performance of sentence representations.   
iv. The trade off between the alignment and uniformity indicates that perfect alignment and perfect uniformity are likely hard to simultaneously achieve in practice.  