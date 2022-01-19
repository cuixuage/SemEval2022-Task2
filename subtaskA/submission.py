from preprocess import *

# 1. Zero-shot
params = {
    'submission_format_file' : '/home/admin/cuixuange/SemEval_2022/SemEval_2022_Task2-idiomaticity/SubTaskA/Data/eval_submission_format.csv' ,
    'input_file'             : '/home/admin/cuixuange/SemEval_2022/SemEval_2022_Task2-idiomaticity/SubTaskA/Data/eval.csv'                   ,
    'prediction_format_file' : '/home/admin/cuixuange/SemEval_2022/TASK1/Models/ZeroShot/0/eval-eval/test_results.txt'                        ,
    }
params[ 'setting' ] = 'zero_shot'

updated_data = insert_to_submission_file( **params )
write_csv( updated_data, './submission/zero_shot_eval_formated.csv' ) 


# 2. One-Shot
params = {
    'submission_format_file' : '/home/admin/cuixuange/SemEval_2022/SemEval_2022_Task2-idiomaticity/SubTaskA/Data/eval_submission_format.csv' ,
    'input_file'             : '/home/admin/cuixuange/SemEval_2022/SemEval_2022_Task2-idiomaticity/SubTaskA/Data/eval.csv'                   ,
    'prediction_format_file' : '/home/admin/cuixuange/SemEval_2022/TASK1/Models/OneShot/1/eval-eval/test_results.txt'                        ,
    }
params[ 'setting' ] = 'one_shot'

updated_data = insert_to_submission_file( **params )
write_csv( updated_data, './submission/one_shot_eval_formated.csv' ) 


# 3. Combine Zero Shot and One Shot submission files.
params = {
    'submission_format_file' : './submission/zero_shot_eval_formated.csv' ,
    'input_file'             : '/home/admin/cuixuange/SemEval_2022/SemEval_2022_Task2-idiomaticity/SubTaskA/Data/eval.csv' ,
    'prediction_format_file' : '/home/admin/cuixuange/SemEval_2022/TASK1/Models/OneShot/1/eval-eval/test_results.txt' ,
    }
params[ 'setting' ] = 'one_shot'

updated_data = insert_to_submission_file( **params )
write_csv( updated_data, './submission/task2_subtaska.csv' ) 

