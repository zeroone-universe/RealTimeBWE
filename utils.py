import os

def get_wav_paths(paths: list):
    wav_paths=[]
    if type(paths)==str:
        paths=[paths]
        
    for path in paths:
        for root, dirs, files in os.walk(path):
            wav_paths += [os.path.join(root,file) for file in files if os.path.splitext(file)[-1]=='.wav']
                        
    wav_paths.sort(key=lambda x: os.path.split(x)[-1])
    
    return wav_paths

def check_dir_exist(path_list):
    if type(path_list) == str:
        path_list = [path_list]
        
    for path in path_list:
        if type(path) == str and os.path.splitext(path)[-1] == '' and not os.path.exists(path):
            os.makedirs(path)       

def get_filename(path):
    return os.path.splitext(os.path.basename(path))  

