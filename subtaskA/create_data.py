from preprocess import *

outpath = 'Data'
    
Path( os.path.join( outpath, 'ZeroShot' ) ).mkdir(parents=True, exist_ok=True)
Path( os.path.join( outpath, 'OneShot' ) ).mkdir(parents=True, exist_ok=True)

# 参数: 输入路径, 输出路径
create_data( '../SemEval_2022_Task2-idiomaticity/SubTaskA/Data/', outpath )

# conda activate hugging_trans
# cd /home/admin/cuixuange/SemEval_2022/TASK1
# python create_data.py