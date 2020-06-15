import  torch
class_num=4
batch_size = 8
def to_one_hot(class_num,batch_size,label):
    """
    将一维列表转换为独热编码
    """
    label = label.resize_(batch_size, 1)
    m_zeros = torch.zeros(batch_size, class_num)
    # 从 value 中取值，然后根据 dim 和 index 给相应位置赋值
    onehot = m_zeros.scatter_(1, label, 1)  # (dim,index,value)

    return onehot.numpy()  # Tensor -> Numpy


label = torch.LongTensor(batch_size).random_() % class_num  # 对随机数取余
print(to_one_hot(class_num,batch_size,label))