# source /etc/profile && source /home/admin/.bash_profile
# source ~/.bashrc
# conda activate hugging_trans
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/miniconda3/lib/
export DATA_DIR="/home/admin/cuixuange/SemEval_2022/TASK1/Data/ZeroShot"
export OUTOUT_DIR="/home/admin/cuixuange/SemEval_2022/TASK1/Models/ZeroShot"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# #################################################################  2021.11.14 No Trainer 脚本
# # accelerate launch  /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro_no_trainer_distributed.py \
# python3 /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro_no_trainer_zero_shot.py \
#     --model_name_or_path '/home/admin/cuixuange/SemEval_2022/transformers_huggingface_co_cache_dir/infoxlm-base' \
#     --max_length 128 \
#     --pad_to_max_length \
#     --gradient_accumulation_steps 1 \
#     --per_device_train_batch_size 16 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 100 \
#     --output_dir $OUTOUT_DIR/0 \
#     --seed 0 \
#     --train_file      $DATA_DIR/train.csv \
#     --validation_file $DATA_DIR/dev.csv \



# ################################################################  2021.11.14 Trainer 脚本
# # 保留Top3 F1指标的CKPT, 其中CKPT目录trainer_statr.json
# python /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro.py \
#     --model_name_or_path '/home/admin/cuixuange/SemEval_2022/transformers_huggingface_co_cache_dir/infoxlm-base' \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 32 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 50 \
#     --evaluation_strategy "epoch" \
#     --output_dir $OUTOUT_DIR/0/ \
#     --seed 0 \
#     --train_file      $DATA_DIR/train.csv \
#     --validation_file $DATA_DIR/dev.csv \
#     --evaluation_strategy "epoch" \
#     --save_strategy "epoch"  \
#     --load_best_model_at_end \
#     --metric_for_best_model "f1" \
#     --save_total_limit 3 \


###### 2021.11.11 predict脚本   0/1_classification
export DATA_DIR="/home/admin/cuixuange/SemEval_2022/TASK1_EVAL/Data/ZeroShot"
python /home/admin/cuixuange/SemEval_2022/AStitchInLanguageModels/Dataset/Task2/Utils/run_glue_f1_macro_no_trainer_zero_shot.py \
    --model_name_or_path '/home/admin/cuixuange/SemEval_2022/TASK1/Models/ZeroShot/f1_0.764' \
    --max_length 128 \
    --pad_to_max_length \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir $OUTOUT_DIR/0/test-test/ \
    --seed 0 \
    --train_file      $DATA_DIR/train.csv \
    --validation_file $DATA_DIR/test.csv \

# cd /home/admin/cuixuange/SemEval_2022/TASK1