import re
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import sys
import csv
import gzip
import math
import torch
import random
import numpy as np
from sentence_transformers.util import import_from_string, batch_to_device
from tqdm.autonotebook import trange
from typing import List, Dict, Optional, Union, Tuple

from datetime                         import datetime
from torch.utils.data                 import DataLoader
from sklearn.metrics.pairwise         import paired_cosine_distances

from datasets                         import load_dataset
from transformers                     import AutoModelForMaskedLM
from transformers                     import AutoTokenizer

from sentence_transformers            import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.readers    import InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

sys.path.append( '/home/admin/cuixuange/SemEval_2022/TASK2_EVAL/SemEval_2022_Task2-idiomaticity-EVAL/SubTaskB/' )
from SubTask2Evaluator                import evaluate_submission

def load_csv( path ) : 
  header = None
  data   = list()

def load_csv( path ) : 
  header = None
  data   = list()
  with open( path, encoding='utf-8') as csvfile:
    reader = csv.reader( csvfile ) 
    for row in reader : 
      if header is None : 
        header = row
        continue
      data.append( row ) 
  return header, data

def tokenise_idiom( phrase ) :
  return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID'

def is_torch_available() :
    try:
        import torch
        return True
    except ImportError:
        return False

def is_tf_available() :
    try:
        import tensorflow as tf
        return True
    except ImportError:
        return False

def set_seed(seed: int):
    """
    Modified from : https://github.com/huggingface/transformers/blob/master/src/transformers/trainer_utils.py
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

        ## From https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.benchmark = False

        ## Might want to use the following, but set CUBLAS_WORKSPACE_CONFIG=:16:8
        # try : 
        #   torch.use_deterministic_algorithms(True)
        # except AttributeError: 
        #   torch.set_deterministic( True )
        
    if is_tf_available():
        import tensorflow as tf
        tf.random.set_seed(seed)

def write_csv( data, location ) : 
  with open( location, 'w', encoding='utf-8') as csvfile:
    writer = csv.writer( csvfile ) 
    writer.writerows( data ) 
  print( "Wrote {}".format( location ) ) 
  return

def tokenize(tokenizer , texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
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

def update_features(sen_a_emb, sen_b_emb, tokens_masks):
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

def get_similarities_extra_cross_layer(sentences1, sentences2, model, batch_size=8):
  ### 2021.12.31 由于引入cross计算方式. 需要传入两个句子,修改Evaluator为分步计算
  model.eval()
  model.to(torch.device("cuda"))
  embeddings_1 = []
  embeddings_2 = []
  assert(len(sentences1) == len(sentences2))
  tokenizer_model = model[0].tokenizer
  # 2022.1.2 print(tokenizer_model)
  ender_model = model[0]
  cross_model = model[1]
  pooling_model = model[-1]
  for start_index in trange(0, len(sentences1), batch_size, desc="Batches", disable=True):
      ########## 1. 分词
      sentences_batches = [sentences1[start_index:start_index+batch_size], sentences2[start_index:start_index+batch_size]]
      input_features = [tokenize(tokenizer_model, s_batch) for s_batch in sentences_batches]
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
      samples = update_features(sen_a_emb, sen_b_emb, tokens_masks)
      sentence_emb = [pooling_model(sample)['sentence_embedding'] for sample in samples]
      # 2022.1.2 print(sentence_emb[0].size(), sentence_emb[1].size())
      embeddings_1.append(sentence_emb[0].detach().cpu())
      embeddings_2.append(sentence_emb[1].detach().cpu())

  embeddings1 = torch.cat(embeddings_1, dim=0).detach().cpu().numpy()
  embeddings2 = torch.cat(embeddings_2, dim=0).detach().cpu().numpy()
  return embeddings1, embeddings2



seed = 4 ## Found using 5 different seeds - specific to this experiment. 
set_seed( seed ) 

data_location = '/home/admin/cuixuange/SemEval_2022/TASK2_EVAL/SemEval_2022_Task2-idiomaticity-EVAL/SubTaskB/EvaluationData/'
outpath = './Models/'
dev_location                = os.path.join( data_location, 'dev.csv'                     ) 
eval_location               = os.path.join( data_location, 'eval.csv'                    ) 
dev_formated_file_location  = os.path.join( data_location, 'dev.submission_format.csv'   ) 
eval_formated_file_location = os.path.join( data_location, 'eval.submission_format.csv'   ) 
test_location                = os.path.join( data_location, 'test.csv'                     ) 
test_formated_file_location = os.path.join( data_location, 'test_submission_format.csv'   ) 
## WARNING: We filter everything based on this (SemEval Task 2 requires that ALL languages are included) 
languages = [ 'EN', 'PT', 'GL' ] 
## Save tmp model here.  
outdir = os.path.join( outpath, 'mBERT' + '-' + str( seed ) ) 
## Save initial Sent Trans model here. 
sent_trans_path  = os.path.join( outpath, 'tokenizedSentTrans_' + str( seed ) )  
## Save final trained model here. 
model_save_path  = os.path.join( outpath, 'tokenizedSentTransNoPreTrain_' + str( seed ) ) 


