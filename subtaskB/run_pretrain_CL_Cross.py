"""
    Nils Reimers has implemented a sentence-transformers-based training code for SimCSE
    https://github.com/princeton-nlp/SimCSE

    export LD_LIBRARY_PATH=$HOME/miniconda3/lib/:$LD_LIBRARY_PATH
"""

from torch.utils.data import DataLoader
import math
import sys
from sentence_transformers import models
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from my_losses import MultipleNegativesRankingLoss_alpha, MultipleNegativesRankingLoss_MoCo, AlignUniformLoss, CosineSimilarityCrossAttentionLoss
from my_evaluators import EmbeddingCrossSimEvaluator
from my_models import CrossAttention, MoCoBuilder
import logging
from datetime import datetime
import os
import gzip
import csv
from utils import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# # ## 1.Extract idioms from Dev and Eval splits
# idioms = list()
# for data_split in [ 'dev', 'eval' , 'test'] : 
#     file_path = os.path.join( data_location, data_split + '.csv' )
#     header, data = load_csv( file_path )
#     for elem in data : 
#         if not elem[ header.index( 'Language' ) ] in languages :
#             continue
#         idioms.append( elem[ header.index( 'MWE1' ) ] )
#         idioms.append( elem[ header.index( 'MWE2' ) ] )
# idioms = list( set( idioms ) ) 
# idioms.remove( 'None' ) 
# print( "Found a total of {} idioms".format( len( idioms ) ) )
# idioms = [ tokenise_idiom( i ) for i in idioms ]            #tokenise_idiom, 添加"ID前缀、ID后缀"重新编码MEW

# ## 2.Download and tokenize model
# model_checkpoint = '/home/admin/cuixuange/SemEval_2022/transformers_huggingface_co_cache_dir/bert-base-multilingual-cased' 
# if languages == [ 'EN' ] :
#     print( "WARNING: Training BERT only instead of mBERT" )
#     model_checkpoint = '/home/admin/cuixuange/SemEval_2022/transformers_huggingface_co_cache_dir/bert-base-cased' 

# model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# model.save_pretrained( outdir )
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, truncation=True)
# tokenizer.save_pretrained( outdir )
# print( "Wrote to: ", outdir, flush=True )

# model          = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
# tokenizer      = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, truncation=True)
# old_len        = len( tokenizer )
# num_added_toks = tokenizer.add_tokens( idioms ) 
# print( "Old tokenizer length was {}. Added {} new tokens. New length is {}.".format( old_len, num_added_toks, len( tokenizer ) )  ) 
# model.resize_token_embeddings(len(tokenizer))
# model.save_pretrained    ( outdir )         ## 2021.11.22  model.bin + config.json
# tokenizer.save_pretrained( outdir )         ## 2021.11.22  vocab + tokens*json
# ## Make sure this worked.   （Test func）
# print( tokenizer.tokenize('This is a IDancienthistoryID'), flush=True )
# print( tokenizer.tokenize( 'This is a IDcolégiomilitarID' ) )

#### 1.DataSet Read
sts_dataset_path = './Datasets/stsbenchmark.tsv.gz'
train_samples_unsup = []
dev_samples   = []
test_samples  = []
train_samples_sup = []
if 'EN' in languages : 
    with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)
            inp_example_1 = InputExample(texts=[row['sentence1'], row['sentence1']])      ###2021.12.1 对比学习，正例的构建是dropout。
            inp_example_2 = InputExample(texts=[row['sentence2'], row['sentence2']])
            if row['split'] == 'dev':
                dev_samples.append(inp_example)
            elif row['split'] == 'test':
                test_samples.append(inp_example)
            else:
                train_samples_unsup.append(inp_example_1)
                train_samples_unsup.append(inp_example_2)
                train_samples_sup.append(inp_example)
            
if 'PT' in languages : 
    for split in [ 'train', 'validation', 'test' ] :
        dataset = load_dataset( path='./utils_assin2.py', split=split )  #2021.10.16 注释，数据使用本地数据时，指定附有数据处理的python文件。
        for elem in dataset :
            score = float( elem['relatedness_score'] ) / 5.0
            inp_example = InputExample(texts=[elem['hypothesis'], elem['premise']], label=score)
            inp_example_1 = InputExample(texts=[elem['hypothesis'], elem['hypothesis']])      ###2021.12.1 对比学习，正例的构建是dropout。
            inp_example_2 = InputExample(texts=[elem['premise'], elem['premise']])
            if split == 'validation':
                dev_samples.append(inp_example)
            elif split == 'test':
                test_samples.append(inp_example)
            elif split == 'train' :
                train_samples_unsup.append(inp_example_1)
                train_samples_unsup.append(inp_example_2)
                train_samples_sup.append(inp_example)
            else :
                raise Exception( "Unknown split. Should be one of ['train', 'test', 'validation']." )


#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
disc = 'mbert_bs32_joint_simcse_and_moco'
outpath = './Models/'
outdir = os.path.join( outpath, 'mBERT' + '-' + str( seed ) ) 
model_path = outdir
sent_trans_path  = os.path.join( outpath, 'tokenizedSentTrans_' + str( seed ) + disc )  
model_save_path  = os.path.join( outpath, 'tokenizedSentTransNoPreTrain_' + str( seed ) + disc) 

#### 1.Use Huggingface/transformers mapping tokens to embeddings
word_embedding_model = models.Transformer(model_path, max_seq_length=64)    #2021.10.16 m-bert模型 + new_tokens100个idioms
for i in word_embedding_model.state_dict():
    print("E_paramter: ", i)
print("train_samples_unsup=", len(train_samples_unsup), " ,dev_samples=", len(dev_samples), " ,test_samples=", len(test_samples))

#### 2. Cross Encoder (need pair-sentence)
cross_model = CrossAttention.CrossAttention()
for i in cross_model.state_dict():
    print("C_paramter: ", i)

#### 3. MoCo Queue
encoder_k_model = models.Transformer(model_path, max_seq_length=64)
moco_queue_model = MoCoBuilder.MoCoBuilder(encoder_q=word_embedding_model, encoder_k=encoder_k_model)

#### 4.Apply mean pooling
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                pooling_mode_mean_tokens=True,
                                pooling_mode_cls_token=False,
                                pooling_mode_max_tokens=False)
tokenizer      = AutoTokenizer.from_pretrained(
    model_path             , 
    use_fast       = False ,
    max_length     = 510   , 
    force_download = True
)
for i in pooling_model.state_dict():
    print("P_paramter: ", i)

model = SentenceTransformer(modules=[word_embedding_model, cross_model, moco_queue_model, pooling_model])
model._first_module().tokenizer = tokenizer
model.save( sent_trans_path )

# Configure the training
train_batch_size =  32
num_epochs = 30

# Use MultipleNegativesRankingLoss for SimCSE
train_dataloader_sup = DataLoader(train_samples_sup, shuffle=True, batch_size=train_batch_size)
train_loss_sup = CosineSimilarityCrossAttentionLoss.CosineSimilarityCrossAttentionLoss(model)   # 2022.1.2  Sentence-Bert-Add-Extra-CrossAttention Loss
loss_weight = 0.15
train_dataloader_unsup = DataLoader(train_samples_unsup, shuffle=True, batch_size=train_batch_size)
train_loss_unsup = MultipleNegativesRankingLoss_alpha.MultipleNegativesRankingLoss(model, loss_weight=loss_weight)  # 2021.12.03 Joint 联合训练, unsup_weight=0.15          
train_loss_AU = AlignUniformLoss.AlignUniformLoss(model, loss_weight=loss_weight)                                   # 2021.12.03 Joint 联合训练, Align&Uniform_weight=0.15
train_loss_simcse_moco = MultipleNegativesRankingLoss_MoCo.MultipleNegativesRankingLoss(model, loss_weight=loss_weight)

warmup_steps = math.ceil(len(train_dataloader_unsup) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

logging.info("Performance before training")
dev_evaluator = EmbeddingCrossSimEvaluator.EmbeddingCrossSimEvaluator.from_input_examples(dev_samples, batch_size=8, main_similarity=None, name='sts-dev')
dev_evaluator(model)

# # Train the model
# model.fit(train_objectives=[(train_dataloader_sup, train_loss_sup),(train_dataloader_unsup, train_loss_simcse_moco)],
#           evaluator=dev_evaluator,
#           epochs=num_epochs,
#           evaluation_steps=250,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path,
#           save_best_model=True
#           )

##############################################################################
# Load the stored model and evaluate its performance on STS benchmark dataset
##############################################################################
model_save_path = "./Models/tokenizedSentTransNoPreTrain_4mbert_bs32_joint_simcse_and_moco"
load_model = SentenceTransformer(model_save_path)
for module in load_model:
    print("module: ", module)
# load_model = model
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator = EmbeddingCrossSimEvaluator.EmbeddingCrossSimEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(load_model, output_path=model_save_path)

