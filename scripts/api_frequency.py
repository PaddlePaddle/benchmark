import os
import json

file = open("API.spec")
cand_api=[]
for line in file.readlines():
    line = line.split()[0]
    subs=['layers', 'contrib', 'optimizer']
    notkeyapi=['data']
    for sub in subs:
        if sub in line:
            line = line.split('.')[-1]
            if line not in notkeyapi:
                cand_api.append(line)

# find models with train.py
path='.'
model_ops=dict()

find_op_path=[]
for root, dirs, files in os.walk(path):
    if any('train.py' in s for s in files):

        # is has multi models
        model_name = root.split('\\')[-1]
        #if not any('model' in s for s in dirs):
        #    find_op_path.append(root)

    #if any('model' in s for s in root):
        find_op_path.append(root)


# find model ops
for op_path in find_op_path:
    cand_ops=['layers.', 'contrib.', 'optimizer.']
    #model_name = op_path.split('\/')[-1]
    model_name = os.path.basename(os.path.normpath(op_path))
    model_ops[model_name] = []
    for root, dirs, op_file in os.walk(op_path):
        for sfile in op_file:
            ext_name = os.path.splitext(sfile)[1]
            if ext_name == '.py':
                fpath = os.path.join(root,sfile)
                with open(fpath) as f_op:
                    for line in f_op.readlines():
                        for op in cand_ops:
                            if op in line:
                                line = line.split('(')[0].split('.')[-1]
                                model_ops[str(model_name)].append(line)


model_ops_spec=dict()
all_ops=[]
# filter op according to API spec
for key, values in model_ops.items():
    model_ops_spec[key]=[]
    for v in values:
        if v in cand_api:
            model_ops_spec[key].append(v)
            all_ops.append(v)

c = dict.fromkeys(all_ops, 0)
for op in all_ops:
    c[op] +=1

def getJSONString(lst):
    join = ""
    rs = "{"
    for i in lst:
        rs += join + '"' + str(i[0]) + '":"' + str(i[1]) + '"'
        join = ","
    return rs + "}"

sort_ops = sorted(c.items(), key=lambda d: d[1], reverse=True)
#sorted_json_ops = json.dumps(dict(sort_ops))
new_ops = getJSONString(sort_ops)

with open('sorted_api.json', 'w') as fp:
    json.dump(dict(sort_ops), fp)
with open('model_api.json', 'w') as fp:
    json.dump(model_ops_spec, fp)
