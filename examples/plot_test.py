import pickle

result_dict = pickle.load(open("./result.pkl", "rb"))   # 从输出pickle中读取数据
print(result_dict)

from rqalpha.mod.rqalpha_mod_sys_analyser import plot

plot("./result.pkl", True, None)