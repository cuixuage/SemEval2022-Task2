"""
export LD_LIBRARY_PATH=$HOME/miniconda3/lib/:$LD_LIBRARY_PATH
"""
import logging
from datetime import datetime
import os
import gzip
import csv
from my_losses import MultipleNegativesRankingLoss_alpha, MultipleNegativesRankingLoss_MoCo, AlignUniformLoss, CosineSimilarityCrossAttentionLoss
from my_evaluators import EmbeddingCrossSimEvaluator
from my_models import CrossAttention, MoCoBuilder
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def _parse_train_data( train_data_location, languages, tokenize=True ) :

  header, train_data = load_csv( train_data_location )
  
  train_data_with_labels                 = list()
  train_data_requiring_labels            = list()
  need_predictions_for_train_data_labels = list()
  print(tokenize, train_data_location, languages)
  # ['ID', 'MWE1', 'MWE2', 'Language', 'sentence_1', 'sentence_2', 'sim', 'alternative_1', 'alternative_2']

  skipped = 0 
  for elem in train_data :

    if not elem[ header.index( 'Language' ) ] in languages :
      skipped += 1
      continue

    mwe1          = elem[ header.index( 'MWE1'          ) ] 
    mwe2          = elem[ header.index( 'MWE2'          ) ] 
    
    this_sim      = elem[ header.index( 'sim'           ) ]
    sentence_1    = elem[ header.index( 'sentence_1'    ) ]
    sentence_2    = elem[ header.index( 'sentence_2'    ) ]
    alternative_1 = elem[ header.index( 'alternative_1' ) ]
    alternative_2 = elem[ header.index( 'alternative_2' ) ]

    ## Remove below if you do not want to tokenize with idiom tokens!
    if tokenize : 
      if mwe1 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe1 ), sentence_1, flags=re.I)
        assert replaced != sentence_1
        sentence_1 = replaced
      if mwe2 != 'None' : 
        replaced = re.sub( mwe1, tokenise_idiom( mwe2 ), sentence_2, flags=re.I)
        assert replaced != sentence_2
        sentence_2 = replaced
  
   
    if this_sim != 'None' :
      tmp = float( this_sim ) 
      train_data_with_labels.append( [ sentence_1, sentence_2, this_sim ] ) 
      continue
    train_data_requiring_labels.append( [ sentence_1, sentence_2 ] ) 
    need_predictions_for_train_data_labels.append( [ alternative_1, alternative_2 ] )

  assert len( need_predictions_for_train_data_labels ) == len( train_data_requiring_labels )
  assert len( train_data ) == len( need_predictions_for_train_data_labels ) + len( train_data_with_labels ) + skipped

  return train_data_with_labels, train_data_requiring_labels, need_predictions_for_train_data_labels 

def _get_predictions_for_train_data_labels( model_path, data ) :

  model      = SentenceTransformer( model_path )

  sentences1 = [ i[0] for i in data ]
  sentences2 = [ i[1] for i in data ]

  # ##  origin
  # embeddings1 = model.encode(sentences1, show_progress_bar=True, convert_to_numpy=True)
  # embeddings2 = model.encode(sentences2, show_progress_bar=True, convert_to_numpy=True)

  ## 2021.1.3
  print(len(sentences1) , sentences1[0], sentences2[0])
  embeddings1, embeddings2 = get_similarities_extra_cross_layer(sentences1, sentences2, model)

  cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

  return cosine_scores

def prepare_eval_data( location, languages, test_print=False ) :
  header, data = load_csv( location )
  sentence1s = list()
  sentence2s = list()
  for elem in data : 
    if not languages is None and not elem[ header.index( 'Language' ) ] in languages : 
      continue
    sentence1 = elem[ header.index( 'sentence1' ) ] 
    sentence2 = elem[ header.index( 'sentence2' ) ] 
    mwe1      = elem[ header.index( 'MWE1'      ) ] 
    mwe2      = elem[ header.index( 'MWE2'      ) ] 

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      print( mwe1 ) 
      print( mwe2 ) 

    if mwe1 != 'None' : 
      replaced = re.sub( mwe1, tokenise_idiom( mwe1 ), sentence1, flags=re.I)
      assert replaced != sentence1
      sentence1 = replaced
    if mwe2 != 'None' : 
      replaced = re.sub( mwe1, tokenise_idiom( mwe2 ), sentence2, flags=re.I)
      assert replaced != sentence2
      sentence2 = replaced

    if test_print : 
      print( sentence1 ) 
      print( sentence2 ) 
      break

    sentence1s.append( sentence1 ) 
    sentence2s.append( sentence2 ) 

  return sentence1s, sentence2s

def generate_train_data( train_data_location, model_path, languages ) :
  
  train_data_with_labels, train_data_requiring_labels, need_predictions_for_train_data_labels = _parse_train_data( train_data_location, languages )
  sims = _get_predictions_for_train_data_labels( model_path, need_predictions_for_train_data_labels )

  train_data_requiring_labels_with_labels = list()
  for index in range( len( train_data_requiring_labels ) ) : 
    train_data_requiring_labels_with_labels.append( [ train_data_requiring_labels[index][0], train_data_requiring_labels[index][1], sims[index] ] )

  train_data = [ [ 'sentence_1', 'sentence_2', 'sim' ] ] + train_data_with_labels + train_data_requiring_labels_with_labels
  assert all( [ (len(i) == 3) for i in train_data ] )
  
  return train_data

def get_similarities( location, model, languages=None ) : 
  sentences1, sentences2 = prepare_eval_data( location, languages ) 
  #Compute embedding for both lists
  
  # ##  origin
  # embeddings1 = model.encode(sentences1, show_progress_bar=True, convert_to_numpy=True)
  # embeddings2 = model.encode(sentences2, show_progress_bar=True, convert_to_numpy=True)

  ## 2021.1.3
  print(len(sentences1) , sentences1[0], sentences2[0])
  embeddings1, embeddings2 = get_similarities_extra_cross_layer(sentences1, sentences2, model)

  # Compute cosine-similarits
  cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

  return cosine_scores


def insert_to_submission( languages, settings, sims, location ) : 
  header, data = load_csv( location ) 
  sims = list( reversed( sims ) )
  ## Validate with length
  updatable = [ i for i in data if i[ header.index( 'Language' ) ] in languages and i[ header.index( 'Setting' ) ] in settings ]
  assert len( updatable ) == len( sims ) 

  ## Will update in sequence - if data is not in sequence must update one language / setting at a time. 
  started_update = False
  for elem in data : 
    if elem[ header.index( 'Language' ) ] in languages and elem[ header.index( 'Setting' ) ] in settings : 
      sim_to_insert = sims.pop()
      elem[-1] = sim_to_insert
      started_update = True
    else :  
      assert not started_update ## Once we start, we must complete. 
    if len( sims ) == 0 : 
      break 
  assert len( sims ) == 0 ## Should be done here. 

  return [ header ] + data ## Submission file must retain header. 


seed   = 1 ## Found using multiple runs
epochs = None ## Found using multiple runs
best_pre_train_seed = 4 ## Found this by running above (as in pre-train setting) multiple times. 
disc='_mbert_fintune_bs32_joint_simcse_and_moco'
outpath = './Models/'
train_data_location = '/home/admin/cuixuange/SemEval_2022/TASK2_EVAL/SemEval_2022_Task2-idiomaticity-EVAL/SubTaskB/TrainData/train_data.csv'
out_location        = './Models/FineTune' + disc + '/'
sent_trans_path  = os.path.join( outpath, 'tokenizedSentTrans_' + str( best_pre_train_seed ) )  
model_path = sent_trans_path + 'mbert_bs32_joint_simcse_and_moco'       # 2021.10.17 tokenizedSentTrans_4
train_data = generate_train_data( train_data_location, model_path, languages )


def create_moco_encoder(model_path):
  """
    MoCoBuilder.py, 其中Load函数没有写好。这里分阶段加载模型。
  """
  word_embedding_model = models.Transformer(model_path, max_seq_length=64)    #2021.10.16 m-bert模型 + new_tokens100个idioms
  cross_model = CrossAttention.CrossAttention()
  encoder_k_model = models.Transformer(model_path, max_seq_length=64)
  moco_queue_model = MoCoBuilder.MoCoBuilder(encoder_q=word_embedding_model, encoder_k=encoder_k_model)
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
  model = SentenceTransformer(modules=[word_embedding_model, cross_model, moco_queue_model, pooling_model])
  model._first_module().tokenizer = tokenizer
  return model

def create_and_eval_subtask_b_fine_tune( model_path, seed, data_location, dev_formated_file_location,eval_formated_file_location,
                                        train_data, out_location, languages, epoch=None, disc = '') :
  set_seed( seed )
  
  dev_location                = os.path.join( data_location, 'dev.csv'                     ) 
  eval_location               = os.path.join( data_location, 'eval.csv'                    ) 


  ## Training Dataloader
  train_samples = list()
  train_samples_unsup = list()

  header     = train_data[0] ## ['sentence_1', 'sentence_2', 'sim']
  train_data = train_data[1:]
  for elem in train_data :
    score = float( elem[2] )
    inp_example = InputExample(texts=[elem[0], elem[1]], label=score)
    inp_example_1 = InputExample(texts=[elem[0], elem[0]])
    inp_example_2 = InputExample(texts=[elem[1], elem[1]])
    train_samples.append(inp_example)
    train_samples_unsup.append(inp_example_1)
    train_samples_unsup.append(inp_example_2)

  ## Params
  train_batch_size = 32
  loss_weight = 0.15
  only_unsup_alpha = 1
  # 2022.1.11 model            = SentenceTransformer( model_path )
  model = create_moco_encoder(model_path)
  for moudle in model:
    print("moudle=", moudle)

  train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
  train_loss = CosineSimilarityCrossAttentionLoss.CosineSimilarityCrossAttentionLoss(model)
  train_dataloader_unsup = DataLoader(train_samples_unsup, shuffle=True, batch_size=train_batch_size)
  train_loss_unsup = MultipleNegativesRankingLoss_alpha.MultipleNegativesRankingLoss(model, loss_weight=loss_weight)        # 2021.12.03 Joint 联合训练, unsup_weight=0.15
  train_loss_simcse_moco = MultipleNegativesRankingLoss_MoCo.MultipleNegativesRankingLoss(model, loss_weight=loss_weight)

 # Train the model
  dev_sims = eval_sims = results = None
  if epoch is None :
    ## Going to test all epochs - notice we can't use the default evaluator. 
    for epoch in range( 1, 30 ) :
      warmup_steps     = math.ceil(len(train_dataloader) * epoch  * 0.1) #10% of train data for warm-up
      print("Warmup-steps: {}".format(warmup_steps), flush=True)
      
      model_save_path = os.path.join( out_location, str( seed ), str( epoch ) ) 
      model.fit(train_objectives=[(train_dataloader, train_loss), (train_dataloader_unsup, train_loss_simcse_moco)],
                evaluator=None,
                epochs=1,
                evaluation_steps=0,
                warmup_steps=warmup_steps,
                output_path=model_save_path
      )

      dev_sims  = get_similarities( dev_location , model, languages ) 
      eval_sims = get_similarities( eval_location, model, languages )

      ## Create submission file on the development set. 
      submission_data = insert_to_submission( languages, [ 'fine_tune' ], dev_sims, dev_formated_file_location )  
      results_file    = os.path.join( outpath, 'dev.combined_results-' + str( seed ) + disc + '.csv' )
      write_csv( submission_data, results_file )

      ## Evaluate development set. 
      results = evaluate_submission( results_file, os.path.join( data_location, 'dev.gold.csv' ) )

      ## Make results printable. 
      for result in results : 
        for result_index in range( 2, 5 ) : 
          result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[ result_index ]
  
      for row in results : 
        print( '\t'.join( [str(i) for i in row ] ) )
        
      results_file = os.path.join( model_save_path, 'RESULTS_TABLE-dev.pre_train_' + str(epoch) + str( seed ) + disc +'.csv' )    
      write_csv( results, results_file )      

      ## Generate combined output for this epoch.
      submission_data = insert_to_submission( languages, [ 'fine_tune' ], eval_sims, eval_formated_file_location )  
      results_file    = os.path.join( outpath, 'eval.combined_results-' + str( seed ) + '_' + str( epoch ) +  disc +'.csv' )
      write_csv( submission_data, results_file )


  else :
    ## We already know the best epoch and so will use it.
    warmup_steps     = math.ceil(len(train_dataloader) * epoch  * 0.1) #10% of train data for warm-up
    print("Warmup-steps: {}".format(warmup_steps), flush=True)

    model_save_path = os.path.join( out_location, str( seed ), str( epoch ) ) 
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=None,
              epochs=epoch,
              evaluation_steps=0,
              warmup_steps=warmup_steps,
              output_path=model_save_path
    )
    
    dev_sims  = get_similarities( dev_location , model, languages ) 
    eval_sims = get_similarities( eval_location, model, languages )

    ## Create submission file on the development set. 
    submission_data = insert_to_submission( languages, [ 'fine_tune' ], dev_sims, dev_formated_file_location )  
    results_file    = os.path.join( outpath, 'dev.combined_results-' + str( seed ) +  disc +'.csv' )
    write_csv( submission_data, results_file )
    
    ## Evaluate development set. 
    results = evaluate_submission( results_file, os.path.join( data_location, 'dev.gold.csv' ) )
    
    ## Make results printable. 
    for result in results : 
      for result_index in range( 2, 5 ) : 
        result[result_index] = 'Did Not Attempt' if result[result_index] is None else result[ result_index ]
  
    results_file = os.path.join( model_save_path, 'RESULTS_TABLE-dev.pre_train_' + str(epoch) + str( seed ) + disc +'.csv' )    
    write_csv( results, results_file )
    
    submission_data = insert_to_submission( languages, [ 'fine_tune' ], eval_sims, os.path.join( data_location, 'eval.submission_format.csv'   )  )  
    results_file    = os.path.join( outpath, 'eval.fine_tune_results-' + str( seed ) + disc +'.csv' )
    write_csv( submission_data, results_file )

    submission_data = insert_to_submission( languages, [ 'fine_tune' ], eval_sims, eval_formated_file_location )  
    results_file    = os.path.join( outpath, 'eval.combined_results-' + str( seed ) + disc +'.csv' )
    write_csv( submission_data, results_file )


  ## Outside if
  return results


params = {
    'model_path'                  : model_path, 
    'seed'                        : seed, 
    'data_location'               : data_location, 
    'dev_formated_file_location'  : './Models/dev.pre_train_results-4.csv',  ## We can append to this.
    'eval_formated_file_location' : './Models/eval.pre_train_results-4.csv',
    'train_data'                  : train_data , 
    'out_location'                : out_location ,  
    'languages'                   : languages ,
    'epoch'                       : epochs,
    'disc'                        : disc
} 


# results = create_and_eval_subtask_b_fine_tune( ** params ) 
# import pandas as pd
# df = pd.DataFrame(data=results[1:], columns=results[0])
# print(df)