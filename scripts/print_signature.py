import sys
import os
import hashlib

py_dict = {}

def get_py_md5(file_path):
    md5 = None
    if os.path.isfile(file_path):
        f = open(file_path,'rb')
        md5_obj = hashlib.md5()
        md5_obj.update(f.read())
        hash_code = md5_obj.hexdigest()
        f.close()
        md5 = str(hash_code).lower()
    return md5

list_dirs = os.walk(sys.argv[1]) 
for root, dirs, files in list_dirs: 
    for f in files:
        if f.endswith('.py'):
            file_path = '%s/%s' %(root, f)
            md5 = get_py_md5(file_path)
            py_dict[file_path] = md5

for py in py_dict:
    print(py, py_dict[py])
