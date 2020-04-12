from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# 自动分配内存防止keras炸显存

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
