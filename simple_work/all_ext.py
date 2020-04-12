import os
# 根据路径和后缀名过滤文件
def list_all_files(rootdir,exten):
    _files =[]
    file_add =[]
    list_file = os.listdir(rootdir)

    for i in range(len(list_file)):
        path = os.path.join(rootdir,list_file[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path,exten))
        if os.path.isfile(path):
             _files.append(path)
    print(_files)
    for file in _files:
        if file.endswith(exten):
            file_add.append(file)
    return file_add

class FileScanner(object):
    # 根据需要更改
    @staticmethod
    def scan(_dir, file_type='tck'):
        filenames = glob.glob(_dir + '*.' + str(file_type))
        
        return {filename.split('/')[-1].split('.')[0]: filename for filename in filenames}