import numpy as np
from sklearn.metrics import f1_score
import paddle as P
import paddle.fluid as F
import paddle.fluid.layers as L
import paddle.fluid.dygraph as D

from ernie.tokenizing_ernie import ErnieTokenizer
from ernie.modeling_ernie import ErnieModelForSequenceClassification

tokenizer = ErnieTokenizer.from_pretrained("erine-1.0")

BATCH=32
MAX_SEQLEN=300
LR=5e-5
EPOCH=10


def make_data(path):
    data = []
    for i,d in enumerate(open(path)):
        if  i==0:
            continue
        else:
            d = d.strip().split(",")
            text,label = d[3],int(d[-1])
            text_id,_ = tokenizer.encode(text)
            text_id = text_id[:MAX_SEQLEN]
            text_id = np.pad(text_id, [0, MAX_SEQLEN - len(text_id)], mode='constant')  # 对所有句子都补长至300，这样会比较费显存；
            # 0是OOV
            label_id = np.array(label + 1)
            data.append((text_id, label_id))
        return data

data = make_data()
train_data,test_data = data[1000:],data[0:1000]
# 进入动态图
D.guard().__enter__()
erine = ErnieModelForSequenceClassification.from_pretrained("ERINE-base",num_labels=3)
optimizer = F.optimizer.Adam(LR, parameter_list=erine.parameters())

def get_batch_data(data,i):
    d = data[i*BATCH:(i+1)*BATCH]
    feature,label = zip(*d)
    feature = np.stack(feature)  # 将BATCH行样本整合在一个numpy.array中
    label = np.stack(list(label))
    feature = D.to_variable(feature) # 使用to_variable将numpy.array转换为paddle tensor
    label = D.to_variable(label)
    return feature, label

for i in range(EPOCH):
    np.random.shuffle(train_data)
    for j in range(len(train_data)//BATCH):
        feature,label = get_batch_data(data,j)
        loss  = erine(feature,labels=label)
        loss.backward()
        optimizer.minimize(loss)
        erine.clear_gradients()
        if j % 10 == 0:
            print('train %d: loss %.5f' % (j, loss.numpy()))
        # evaluate
        # evaluate
        if j % 100 == 0:
            all_pred, all_label = [], []
            with D.base._switch_tracer_mode_guard_(is_train=False):  # 在这个with域内ernie不会进行梯度计算；
                erine.eval()  # 控制模型进入eval模式，这将会关闭所有的dropout；
                for j in range(len(test_data) // BATCH):
                    feature, label = get_batch_data(test_data, j)
                    loss, logits = erine(feature, labels=label)
                    all_pred.extend(L.argmax(logits, -1).numpy())
                    all_label.extend(label.numpy())
                erine.train()
            f1 = f1_score(all_label, all_pred, average='macro')
            print('f1 %.5f' % f1)


acc=  P.fluid.layers.accuracy()