import numpy as np
import pandas as pd
import json
with open('./q/q.json','r') as f:
    q=json.load(f)
input_array = np.array([[j for j in i.values()] for i in q.values()]).flatten()
out = np.concatenate(input_array).ravel()
print(pd.DataFrame(out).describe())