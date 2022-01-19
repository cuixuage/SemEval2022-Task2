# source /etc/profile && source /home/admin/.bash_profile
# source ~/.bashrc
# conda activate hugging_trans
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
export DATA_DIR="/home/admin/cuixuange/SemEval_2022/TASK1/Data/OneShot"
export OUTOUT_DIR="/home/admin/cuixuange/SemEval_2022/TASK1/Models/OneShot"
export CUDA_VISIBLE_DEVICES=1,2,3

# ####### 2021.11.11 train脚本
# python /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro_no_trainer_one_shot.py \
#     --model_name_or_path '/home/admin/cuixuange/SemEval_2022/transformers_huggingface_co_cache_dir/infoxlm-base' \
#     --max_length 128 \
#     --pad_to_max_length \
#     --gradient_accumulation_steps 1 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 100 \
#     --output_dir $OUTOUT_DIR/1_1_withoutGL/ \
#     --seed 1 \
#     --train_file      $DATA_DIR/train.csv \
#     --validation_file $DATA_DIR/dev.csv \



###### 2021.11.11 predict脚本   0/1_classification
export DATA_DIR="/home/admin/cuixuange/SemEval_2022/TASK1_EVAL/Data/OneShot"
python /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro_no_trainer_one_shot.py \
    --model_name_or_path '/home/admin/cuixuange/SemEval_2022/TASK1/Models/OneShot/f1_0.918' \
    --max_length 128 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir $OUTOUT_DIR/1/test-test/ \
    --seed 1 \
    --train_file      $DATA_DIR/train.csv \
    --validation_file $DATA_DIR/test.csv \


# cd /home/admin/cuixuange/SemEval_2022/TASK1
