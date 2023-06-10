import glob
import random
from PIL import Image
import os

#函数定义
def read_split_data(data_path,ratio):
    test_split_ratio=ratio
    raw_path=data_path

    dirs=glob.glob(os.path.join(raw_path,'*'))
    dirs=[d for d in dirs if os.path.isdir(d)]
    
    print(f'totally{len(dirs)}classes:{dirs}')

    for path in dirs:
        path=path.split('\\')[-1]
        print(path)

        os.makedirs(f'train/{path}',exist_ok=True)
        os.makedirs(f'train/{path}',exist_ok=True)

        files=glob.glob(os.path.join(raw_path,path,'*'))
        
        random.shuffle(files)

        boundary=int(len(files)*test_split_ratio)
        print('len(files):',len(files))
        print('boundary:',boundary)

        for i,file in enumerate(files):
            img=Image.open(file)
            if i<= boundary:
                img.save(os.path.join(f'test/{path}',file.split('//')[-1].split('.')))
            else:
                img.save(os.path.join(f'train/{path}',file.split('//')[-1].split('.')))
    
    test_files =glob.glob(os.path.join('test','*','*.jpg'))
    train_files=glob.glob(os.path.join('train','*','*.jpg'))

    print(f'totally{len(train_files)} files for training')
    print(f'totally{len(test_files)} files for test')


data_path='./fruit30_train'
ratio=0.2
read_split_data(data_path,ratio)