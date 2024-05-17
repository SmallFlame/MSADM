import os  
import shutil  
from itertools import islice  
  
def copy_first_n_files(src_dir, dst_dir, n):  
    # 确保目标文件夹存在  
    if not os.path.exists(dst_dir):  
        os.makedirs(dst_dir)  
  
    # 获取源文件夹中所有文件的列表（不包括子文件夹中的文件）  
    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]  
  
    # 使用islice复制前n个文件  
    for file_path in islice(files, n):  
        # 构造目标文件路径  
        dst_file_path = os.path.join(dst_dir, os.path.relpath(file_path, src_dir))  
  
        # 如果目标文件路径的目录不存在，则创建它  
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)  
  
        # 复制文件  
        shutil.copy2(file_path, dst_file_path)  
  
n = 500  # 要复制的文件数量  
src_dir = 'data/train/train9to12/ruler/appdown'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/appdown'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/congest'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/congest'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/malicious'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/malicious'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/nodedown'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/nodedown'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/normal'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/normal'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/obstacle'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/obstacle'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)
src_dir = 'data/train/train9to12/ruler/out'  # 替换为你的源文件夹路径  
dst_dir = 'data/train/val9to12/ruler/out'  # 替换为你的目标文件夹路径  
copy_first_n_files(src_dir, dst_dir, n)