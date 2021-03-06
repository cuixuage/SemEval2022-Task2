1.问题的定义  
```
    SubTaskA, zero-shot、one-shot; 单句分类、双句分类问题。[CLS]SentenceA[SEP]SentenceB[SEP]    
    SubTaskB, pre-train、fine-tune; 句子向量的质量评估。sentence_emb=avg(tokens_embed)   
    备注: avg平均的方式使得句子向量容易被高频Tokens语义主导，可能存在"坍缩"现象。
```  

2.方法调研(2021.10)  
2.1 多语言模型  
```
    Xtreme Leader-Board; M-Bert, XLM-R, InfoXLM  
```

2.2 少样本学习  
```
    Prompt, 看了NLPCC 2021少样本比赛, Top方法和自动生成prompt不相关, 暂不使用。  
```

2.3 数据增强  
```
    对抗训练: FGM\PGD\FreeLB\Smart  
    MixUp:   
    MoCo: Queue保存历史embed信息, 扩充负样本  
    https://tech.meituan.com/2021/08/19/low-resource-learning.html  
```

2.4 Sentence向量表征  
```
    缺点:  
        sentence-bert双塔模型直接使用avg_tokens_emb作为句子向量表征，但是两个句子相似度计算前没有交互信息。  
    我们提议使用:  
        i. 添加CrossAttention。对于两个句子的token embedding加强交互信息，相比于直接使用MeanPooling计算Cosine效果可以进一步提升。（Paper: Sentence-Bert.2019）  
        ii. 添加辅助对比学习函数。拉大正例、负例之间的Sentence_Embedding距离（Paper: ConSert.2021）  
            a. InfoNCE损失。注意softmax分母是1个正例，batch_size-1个负例构成。（Paper: SimCSE.2021）  
            b. 增加负例的样本数。提高hard-negative出现的概率，使用Queue保存历史Embedding。（Paper: MoCo.2019）  
```

2.4 定稿  
```
    InfoXLM + EMA + Adversarial Training + Contrastive Learning   
``` 

3.效果验证  
```
    第五章节，消融实验、可视化分析
```

4.收获&疑惑     
4.1 多语言模型  
```
    a. 数据预处理  
        i. 分词工具的选择；  BPE，WordPiece，SentencePiece    
        ii. Vocab构建前需要对于数据重新采样； 平衡rich-resource，low-resource语种    
    b. 预训练任务  
        i. MMLM  
        ii. TLM  
        iii. XLCo  
    c. 适用边界  
        如此多的预训练任务,分别适用于哪些场景?    
``` 
   
4.2 CrossAttention  
```
    a.Encoder输出token-level hidden-status，后续加上额外的非线性变换模块，有助于hidden-status学习更好。  
    b.Predict/Eval阶段，不使用CrossAttention效果也会变好。  
```
    
4.3 对比学习    
```
    a. unsup-simcse相比于m-bert，通过降低Alignment带来了Uniformity的提升  
    b. 引入Cross Attention网络、增加对比损失函数能够增加Embedding的Uniformity，但是会带来Alignment指标的损失  
   
    i. 正例、负例的构造  
    ii. Encoder映射函数的设计   
        a. 对于隐层向量加上MLP做非线性变换，用于增强hidden status的表征能力。(SimCLR)   
    iii. Loss损失函数的定义   
        a. 负例抽取的决策点。SimCLR = in-batch negative samples； MoCo = history Queue batch negative samples  
        b. 语义相似度的决策点。Similarity = Consine Sim = L2 Norm + Dot Product   
        c. 温度系数的决策点。 0.01 ~ 0.1更小的T更容易聚焦到困难样本Loss   
``` 