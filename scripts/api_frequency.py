import os
import json
import re
import string
import pymysql

def findheader(path):
    name_2_model=dict()
    for root in os.listdir(path):
        if root.find('.py') != -1:
            with open(os.path.join(path, root)) as f:
                for line in f.readlines():
                    if (line.startswith('from') or line.startswith('import')) and "models." in line and not "check" in line:
                        string = line.split()[1].rsplit('.')[1]
                        model_path = os.path.join(path, '../models', string)
                        model_name = os.path.basename(os.path.normpath(path))
                        name_2_model[model_name]=model_path
			print(model_name, model_path)
                    if line.startswith('sys.path.append') and 'models/*' in line:
                        string = line.split('/')[-2]
                        model_path = os.path.join(path, '../models', string)
                        model_name = os.path.basename(os.path.normpath(path))
                        name_2_model[model_name]=model_path
                        
    return name_2_model
                
path=os.environ.get('MODEL_PATH')
file = open(path+"/API.spec")
cand_api=[]

for line in file.readlines():
    line = line.split()[0]
    subs=['layers', 'contrib', 'optimizer', 'metrics']
    notkeyapi=['data', 'backward', '__init__', 'log', 'input', 'output', 'run', 'ls', 'train', 'shape', 'eval', 'sample', 'decode']
    for sub in subs:
        if sub in line:
            line = line.split('.')[-1]
            if line not in notkeyapi:
                cand_api.append(line)

cand_api=list(set(cand_api))
# find models with train.py
model_ops=dict()

know_models=[]
find_op_path=[]
cv_sub_models=['PaddleDetection', 'face_detection', 'human_pose_estimation',
'image_classification', 'metric_learning', 'ocr_recognition',
'PaddleGAN/network', 'PaddleVideo/models', 'PaddleSeg', 'Research']

nlp_sub_models=['PaddleDialogue', 'PaddleLARK', 'PaddleMRC', 'PaddleMT',
'PaddleTextGEN', 'dialogue_domain_classification', 'Research']

nlp_refer_models=['emotion_detection', 'language_model', 'lexical_analysis',
'sentiment_classification', 'similarity_net']

rec_sub_models=['ctr', 'din', 'gnn', 'gru4rec', 'multiview_simnet',
'tagspace', 'ssr', 'text_matching_on_quora', 'word2vec']

speech_sub_models=['DeepASR', 'DeepVoice3', 'DeepSpeech']

for tops in os.listdir(path):
    if tops == "PaddleCV":
        for root in os.listdir(path+tops):
            sub_path = os.path.join(path, tops, root)
            if root in cv_sub_models:
                find_op_path.append(sub_path)
    elif tops == 'PaddleNLP':
        real_model_path=[]
        for root in os.listdir(path+tops):
            sub_path = os.path.join(path, tops, root)
            if root in nlp_sub_models:
                find_op_path.append(sub_path)
            elif root in nlp_refer_models:
                #this model's path in real model path list
                name_path = findheader(sub_path)
                real_model_path.append(name_path)
                find_op_path.append(sub_path)
    elif tops == 'PaddleRec':
        for root in os.listdir(path+tops):
            sub_path = os.path.join(path, tops, root)
            if root in rec_sub_models:
                find_op_path.append(sub_path)
    elif tops == 'PaddleSlim':
        sub_path = os.path.join(path, tops)
        find_op_path.append(sub_path)
    elif tops == 'PaddleKG':
        sub_path = os.path.join(path, tops)
        find_op_path.append(sub_path)
    elif tops == 'PaddleSpeech':
        for root in os.listdir(path+tops):
            sub_path = os.path.join(path, tops, root)
            if root in speech_sub_models:
                find_op_path.append(sub_path)

# declare alias name
alias_name=dict()
alias_name["Adam"]="AdamOptimizer"
alias_name["SGD"]="SGDOptimizer"
alias_name["Momentum"]="MomentumOptimizer"
alias_name["SGD"]="SGDOptimizer"
alias_name["Ftrl"]="FtrlOptimizer"
alias_name["DecayedAdagrad"]="DecayedAdagradOptimizer"
alias_name["Adadelta"]="AdadeltaOptimizer"
alias_name["Adagrad"]="AdagradOptimizer"
alias_name["Adamax"]="AdamaxOptimizer"
alias_name["LarsMomentum"]="LarsMomentumOptimizer"

find_op_path=list(set(find_op_path))

# find ops used in model path
for op_path in find_op_path:
    cand_ops=['layers.', 'contrib.', 'optimizer.', 'metrics.']
    #model_name = op_path.split('\/')[-1]
    model_name = os.path.basename(os.path.normpath(op_path))
    model_ops[model_name] = []
    for name_path in real_model_path:
        if model_name in name_path.keys():
            op_path = name_path.values()[0]
        
    for root, dirs, op_file in os.walk(op_path):
        for sfile in op_file:
            ext_name = os.path.splitext(sfile)[1]
            if ext_name == '.py':
                fpath = os.path.join(root,sfile)
                with open(fpath) as f_op:
                    for line in f_op.readlines():
                        for op in cand_ops:
                            if op in line:
                                #line = line.split('(')[0].split('.')[-1]
                                line = line.split(op)[-1]
				#op_name = re.sub("[0-9,a-z,A-Z,_].*", "", line)
				if line.find('('):
				    line = line.split('(')[0]
                                op_name = filter(lambda x: x in string.ascii_letters + string.digits + '_', line)
				if op_name != "":
   				    if op_name in alias_name.keys() and alias_name[op_name]:
				        op_name = alias_name[op_name]
                                    model_ops[str(model_name)].append(op_name)

model_ops_spec=dict()
all_ops=[]

# sql
db = pymysql.connect(host='gzbh-qianmo-com-162-69-149.gzbh.baidu.com', port=3306, user='root', passwd='', db='paddle', charset='utf8')
cursor = db.cursor()
table_name='model_op_frequency'
cursor.execute("select * from %s" % table_name)
db.commit()
#cursor.execute("TRUNCATE TABLE %s" % table_name)
#cursor.execute("select * from %s" % table_name)
col_name_list = [tuple[0] for tuple in cursor.description]

sql = "INSERT INTO model_op_frequency (model, op, count) VALUES ( '%s', '%s', %d )"
# filter op according to API spec
for key, values in model_ops.items():
    model_ops_spec[key]=[]
    # remove duplicated op
    #new_values = set(values)
    model_ops=[]
    for v in values:
        if v in cand_api:
            model_ops_spec[key].append(v)
            all_ops.append(v)
	    model_ops.append(v)
    op_count=dict.fromkeys(model_ops, 0)
    for op in model_ops:
	op_count[op] +=1
    for op_name, counts in op_count.items():
        print(key, op_name, counts)
	#cursor = db.cursor()
        #data=(key, op_name, counts)
	#cursor.execute(sql % data)
	#db.commit()

# construct a dict from op name
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
print(len(sort_ops))
#sorted_json_ops = json.dumps(dict(sort_ops))
new_ops = getJSONString(sort_ops)

with open('sorted_api.json', 'w') as fp:
    json.dump(sort_ops, fp)
with open('model_api.json', 'w') as fp:
    json.dump(model_ops_spec, fp)
