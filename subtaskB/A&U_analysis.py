from torch.utils.data import DataLoader
import torch.nn.functional as F
from sentence_transformers import models
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,SimilarityFunction
import csv
from datasets                         import load_dataset
import os, random, gzip, logging
import torch
from transformers import (
    default_data_collator,
    AutoTokenizer,
    AutoModel
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
REF:
    https://github.com/SsnL/align_uniform
    https://github.com/princeton-nlp/SimCSE/issues/85
    https://github.com/princeton-nlp/SimCSE/issues/41
"""
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def align_loss(x, y, alpha=2):    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def get_embedding(model, batch):
    """
        Note:
        1. unsup-simcse use last_hidden_state
        2. sup-simcse use pooler_output
    """
    outputs = model(input_ids=batch['input_ids'].cuda(), attention_mask=batch['attention_mask'].cuda(), token_type_ids=batch['token_type_ids'].cuda())
    # print(outputs.keys())   # 2021.12.08  ['last_hidden_state', 'pooler_output']
    last_hidden_state = outputs.last_hidden_state
    # print(last_hidden_state[:,0].shape)     #2021.12.08 [bs, 768]
    return  last_hidden_state[:,0]

def get_align(model, dataloader1, dataloader2):
    align_all = []
    assert(len(dataloader1) == len(dataloader2))
    with torch.no_grad():        
        for batch1, batch2 in zip(dataloader1, dataloader2):
            z1 = get_embedding(model, batch1) 
            z2 = get_embedding(model, batch2)  
            z1 = F.normalize(z1,p=2,dim=1)
            z2 = F.normalize(z2,p=2,dim=1)
            align_all.append(align_loss(z1, z2, alpha=2))     
    return align_all

def get_unif(model, dataloader1, dataloader2):
    unif_all = []
    assert(len(dataloader1) == len(dataloader2))
    with torch.no_grad():        
        for batch1, batch2 in zip(dataloader1, dataloader2):
            z1 = get_embedding(model, batch1) 
            z2 = get_embedding(model, batch2)  
            z1 = F.normalize(z1,p=2,dim=1)
            z2 = F.normalize(z2,p=2,dim=1)
            z = torch.cat((z1,z2))
            unif_all.append(uniform_loss(z, t=2))

    return unif_all

def preprocess_function_1(examples):
    # Tokenize the texts
    result = tokenizer(examples['sentence1'], padding="max_length", max_length=64, truncation=True)
    result["labels"] = examples["float_socre"]      # is_regression
    return result

def preprocess_function_2(examples):
    # Tokenize the texts
    result = tokenizer(examples['sentence2'], padding="max_length", max_length=64, truncation=True)
    result["labels"] = examples["float_socre"]      # is_regression
    return result

def STS_Assign_2_csv(pos_output_file, all_output_file):
    """
        STS + ASSIGN2,  store format as "sent1, sent2, label".csv
    """
    sts_dataset_path = './Datasets/stsbenchmark.tsv.gz'
    pos_dataset = []
    all_dataset = []
    languages = [ 'EN', 'PT' ]
    # languages = [ 'EN' ]            # 2021.12.08 only STS-Benchmark
    if 'EN' in languages : 
        with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                tuple_example = (row['sentence1'], row['sentence2'], row['score'])
                all_dataset.append(tuple_example)
                if float(row['score']) >= 4.0:
                    pos_dataset.append(tuple_example)
                
    if 'PT' in languages : 
        for split in [ 'train', 'validation', 'test' ] :
            dataset = load_dataset( path='./utils_assin2.py', split=split )
            for elem in dataset :
                tuple_example = (elem['hypothesis'], elem['premise'], elem['relatedness_score'])
                all_dataset.append(tuple_example)
                if float(elem['relatedness_score']) >= 4.0:
                    pos_dataset.append(tuple_example)
    print("len(pos_dataset)=", len(pos_dataset), " len(all_dataset)=", len(all_dataset))

    header = ['sentence1', 'sentence2', 'float_socre']
    with open(pos_output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter ='\t')
        writer.writerow(header)
        for item in pos_dataset:  writer.writerow(list(item))

    with open(all_output_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter ='\t')
        writer.writerow(header)
        for item in all_dataset:  writer.writerow(list(item))

def process_csv(pos_output_file):
    num_labels = 1   # is_regression
    raw_pos_datasets = load_dataset('csv', data_files={'test': pos_output_file}, delimiter='\t')
    processed_pos_datasets_1 = raw_pos_datasets.map(
        preprocess_function_1,
        batched=True,
        remove_columns=['sentence2', 'float_socre'],
        desc="Running tokenizer on dataset",
    )['test']
    processed_pos_datasets_2 = raw_pos_datasets.map(
        preprocess_function_2,
        batched=True,
        remove_columns=['sentence1', 'float_socre'],
        desc="Running tokenizer on dataset",
    )['test']
    # Log a few random samples from the training set:
    for index in range(0,3):
        logging.info(f"Sample {index} of the set——1: {processed_pos_datasets_1[index]}.")
    for index in range(0,3):
        logging.info(f"Sample {index} of the set——2: {processed_pos_datasets_2[index]}.")
    return processed_pos_datasets_1, processed_pos_datasets_2


#### 1. load_model
model_name_or_path = './Models/tokenizedSentTransNoPreTrain_4mbert_bs32_joint_simcse_and_moco'
# tokenizedSentTransNoPreTrain_4mbert_joint_bs32   tokenizedSentTransNoPreTrain_4mbert_unsup_simcse_bs32    tokenizedSentTransNoPreTrain_4mbert_bs32_sup  tokenizedSentTransNoPreTrain_4mbert_bs32_sup_cross_train_and_eval  tokenizedSentTransNoPreTrain_4mbert_bs32_joint_simcse_and_cross_train_and_eval
model = AutoModel.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = model.cuda()


#### 2. load_data + tokenization
all_output_file = './Datasets/A&U/all_data.csv'
pos_output_file = './Datasets/A&U/pos_data.csv'
STS_Assign_2_csv(pos_output_file, all_output_file)
processed_pos_datasets_1, processed_pos_datasets_2 = process_csv(pos_output_file)
processed_all_datasets_1, processed_all_datasets_2 = process_csv(all_output_file)


####### 2. get_embedding + A&U analysis
batch_size = 64
pos_dataloader_1 = DataLoader(processed_pos_datasets_1, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)
pos_dataloader_2 = DataLoader(processed_pos_datasets_2, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)
all_dataloader_1 = DataLoader(processed_all_datasets_1, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)
all_dataloader_2 = DataLoader(processed_all_datasets_2, shuffle=False, collate_fn=default_data_collator, batch_size=batch_size)

align_all = get_align(model, pos_dataloader_1, pos_dataloader_2)
align = sum(align_all)/len(align_all)
print("Align=", align.cpu())

unif_all = get_unif(model, all_dataloader_1, all_dataloader_2)
unif = sum(unif_all)/len(unif_all)
print("Uniform=", unif.cpu())



"""
1.STS-B 
unsup-simcse-bert-base-uncased: Align=0.2321, Uniform=-2.3106   # similaer to the official results

2.STS-B + Assign2
unsup-simcse-bert-base-uncased: Align=0.1986, Uniform=-1.7282
sup-simcse-bert-base-uncased: Align=0.1109, Uniform=-1.3303

3.STS-B + Assign2
bert-base-multilingual-cased:        Align=0.0755, Uniform=-0.4418
unsup-simcse-bert-base-uncased:      Align=0.1986, Uniform=-1.7282
sup-simcse-bert-base-uncased:        Align=0.1109, Uniform=-1.3303

bert-base-multilingual-cased_sup:    Align=0.1468, Uniform=-2.5506
bert-base-multilingual-cased_unsup:  Align=0.0931, Uniform=-0.6409
bert-base-multilingual-cased_joint:  Align=0.1562, Uniform=-2.9160

bert-base-multilingual-cased_sup_cross:     Align=0.1772, Uniform=-2.6359
bert-base-multilingual-cased_joint_cross:   Align=0.3547, Uniform=-3.2408

bert-base-multilingual-cased_joint_cross_moco:   Align=0.2781, Uniform=-2.8412
"""