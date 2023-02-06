from stable_baselines3 import DQN

# 加载模型
from encapsulation.utils import prehandle, get_obs

model = DQN.load("../IF8888model_DQN_20210514")

# 准备数据
df = prehandle(df)

obs = get_obs(df.tail(1))

model.predict(obs)[0]