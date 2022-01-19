from sentence_transformers.evaluation import SentenceEvaluator,SimilarityFunction
import logging
import torch
import csv
import os
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sentence_transformers.readers import InputExample
from torch._C import device
from typing import List, Dict, Optional, Union, Tuple
from torch.nn.functional import embedding
from tqdm.autonotebook import trange
from sentence_transformers.util import import_from_string, batch_to_device, fullname, snapshot_download

logger = logging.getLogger(__name__)

class EmbeddingCrossSimEvaluator(SentenceEvaluator):
    """
    代码参参考:  sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    区别点:
        1. 通过transformer库API, 获取token-wise embedding
        2. cross_attention from pairs-sentence
        3. pooling the "new" token-wise embedding
    """
    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float], batch_size: int = 16, main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False, write_csv: bool = True):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = main_similarity
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "similarity_evaluation"+("_"+name if name else '')+"_results.csv"
        self.csv_headers = ["epoch", "steps", "cosine_pearson", "cosine_spearman", "euclidean_pearson", "euclidean_spearman", "manhattan_pearson", "manhattan_spearman", "dot_pearson", "dot_spearman"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def tokenize(self, tokenizer , texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
            Tokenizes a text and maps tokens to token-ids
            CodeFrom: sentence_transformers/models/Transformer.py
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        #strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        #Lowercase
        do_lower_case = False   #2022.1.1
        if do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]


        output.update(tokenizer(*to_tokenize, padding=True, truncation='longest_first', return_tensors="pt", max_length=64))
        return output

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

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        ### 2021.12.31 由于引入cross计算方式. 需要传入两个句子,修改Evaluator为分步计算
        model.eval()
        model.to(torch.device("cuda"))
        embeddings_1 = []
        embeddings_2 = []
        assert(len(self.sentences1) == len(self.sentences2))
        tokenizer_model = model[0].tokenizer
        # 2022.1.2 print(tokenizer_model)
        ender_model = model[0]
        cross_model = model[1]
        pooling_model = model[-1]
        for start_index in trange(0, len(self.sentences1), self.batch_size, desc="Batches", disable=True):
            ########## 1. 分词
            sentences_batches = [self.sentences1[start_index:start_index+self.batch_size], self.sentences2[start_index:start_index+self.batch_size]]
            input_features = [self.tokenize(tokenizer_model, s_batch) for s_batch in sentences_batches]
            input_features = [batch_to_device(input_feature, torch.device("cuda")) for input_feature in input_features] ### 2022.1.1 input_ids;attention_mask;token_type_ids

            ########## 2. 获取token-wise embedding
            output_features = [ender_model(features) for features in input_features]
            tokens_emb = [output_fea['token_embeddings'] for output_fea in output_features]
            tokens_masks = [output_fea['attention_mask'] for output_fea in output_features]
            # 2022.1.2 print(tokens_emb[0].size(), tokens_emb[1].size(), tokens_masks[0].size(), tokens_masks[1].size())

            ########## 3. Cross 计算 (Q + KV + Padding_Mask)
            extended_attention_masks = []
            for item in tokens_masks:
                attention_mask = item.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
                extended_attention_masks.append(attention_mask)
            sen_a_emb = cross_model(features=output_features[0], hidden_states=tokens_emb[0],
                                    encoder_hidden_states=tokens_emb[1], encoder_attention_mask=extended_attention_masks[1])['cross_token_embeddings']
            sen_b_emb = cross_model(features=output_features[1], hidden_states=tokens_emb[1],
                                    encoder_hidden_states=tokens_emb[0], encoder_attention_mask=extended_attention_masks[0])['cross_token_embeddings']
            # 2022.1.2 print(sen_a_emb.size(), sen_b_emb.size()) # torch.Size([32, 41, 768]) torch.Size([32, 39, 768])

            ########## 4. Pooling 计算
            samples = self.update_features(sen_a_emb, sen_b_emb, tokens_masks)
            sentence_emb = [pooling_model(sample)['sentence_embedding'] for sample in samples]
            # 2022.1.2 print(sentence_emb[0].size(), sentence_emb[1].size())
            embeddings_1.append(sentence_emb[0].detach().cpu())
            embeddings_2.append(sentence_emb[1].detach().cpu())

        embeddings1 = torch.cat(embeddings_1, dim=0).detach().cpu().numpy()
        embeddings2 = torch.cat(embeddings_2, dim=0).detach().cpu().numpy()
        # print(embeddings1.shape, embeddings2.shape)   # 2022.1.2 (2000, 768) (2000, 768)

        ######## 2022.1.2  sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py
        # embeddings1 = model.encode(self.sentences1, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        # embeddings2 = model.encode(self.sentences2, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        labels = self.scores

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]


        eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
        eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

        eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
        eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

        eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
        eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

        eval_pearson_dot, _ = pearsonr(labels, dot_products)
        eval_spearman_dot, _ = spearmanr(labels, dot_products)

        logger.info("Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_cosine, eval_spearman_cosine))
        logger.info("Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_manhattan, eval_spearman_manhattan))
        logger.info("Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_euclidean, eval_spearman_euclidean))
        logger.info("Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
            eval_pearson_dot, eval_spearman_dot))

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson_cosine, eval_spearman_cosine, eval_pearson_euclidean,
                                 eval_spearman_euclidean, eval_pearson_manhattan, eval_spearman_manhattan, eval_pearson_dot, eval_spearman_dot])


        if self.main_similarity == SimilarityFunction.COSINE:
            return eval_spearman_cosine
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return eval_spearman_euclidean
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return eval_spearman_manhattan
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return eval_spearman_dot
        elif self.main_similarity is None:
            return max(eval_spearman_cosine, eval_spearman_manhattan, eval_spearman_euclidean, eval_spearman_dot)
        else:
            raise ValueError("Unknown main_similarity value")
