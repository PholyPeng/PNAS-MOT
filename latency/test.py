import json
import pickle

with open('./lut.pkl', 'rb') as f:
    latency_dict = pickle.load(f)
    # print(latency_dict['max_pool_3x3'])
    beautiful_format = json.dumps(latency_dict, indent=4, ensure_ascii=False)
    print(beautiful_format)