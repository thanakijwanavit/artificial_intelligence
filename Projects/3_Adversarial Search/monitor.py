import time
import os
import json
while True:
    print('number of q files is {}'.format(len(os.listdir('./q'))))
    with open('./q/q.json','r') as f:
        q=json.load(f)
    print('length of q is {}'.format(len(q.keys())))
    time.sleep(9)