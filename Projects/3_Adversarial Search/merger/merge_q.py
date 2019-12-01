import argparse
import json
import os
import pprint
import time
import numpy as np
def merge_dict_add_value(dict1, dict2):
   ''' Merge dictionaries and keep values of common keys in list'''
   dict3 = {**dict1, **dict2}
   for key, value in dict3.items():
       if key in dict1 and key in dict2:
               dict3[key] = value/2 + dict1[key]/2
               if dict3[key] > 1:
                   dict3[key] = 1
               elif dict3[key] < -1:
                   dict3[key] = -1
   return dict3

files = os.listdir('./q')
files_without_main_q = [file for file in files if file not in ['q.json','q_updated.json']]
if len(files_without_main_q) > 10:
    files_without_main_q = np.array(files_without_main_q)[np.random.randint(0,len(files_without_main_q),10)]
print(files_without_main_q)
if 'q.json' not in os.listdir('./q'):
    with open('./q/q.json','w') as f:
        json.dump({},f)
with open('./q/q.json', 'r') as f:
    q=json.load(f)

print('len of q is {}'.format(len(q.keys())))
    
for qs_name in files_without_main_q:
    try:
        with open('./q/'+qs_name, 'r') as f:
            qs = json.load(f)
        for key in qs.keys():
            if key in q.keys():
                q[key] = merge_dict_add_value(q[key],qs[key])
            else:
                q[key] = qs[key]
        print(' len of qs is {}, len of new q is {}'.format(len(qs.keys()),len(q.keys())))
    except:
        continue
tmp_dir = './q/q_updated{}.json'.format(np.random.randint(100,99999))
with open(tmp_dir, 'w') as f:
    q=json.dump(q,f)

for file_name in files_without_main_q:
    try:
        os.remove('./q/'+file_name)
    except:
        continue
os.rename( tmp_dir , './q/q.json')