# RNN





#Seq2Seq



## Sequence to Sequence模型

<img src="https://pic3.zhimg.com/80/v2-a5012851897f8cc685bc946e73496304_1440w.jpg" alt="img" style="zoom: 33%;" />图5

在Seq2Seq结构中，编码器Encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由解码器Decoder解码。在解码器Decoder解码的过程中，不断地将前一个时刻 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 的输出作为后一个时刻 ![[公式]](https://www.zhihu.com/equation?tex=t) 的输入，循环解码，直到输出停止符为止。

<img src="https://picb.zhimg.com/80/v2-343dbbf86c8e92e9fc8d6b3a938c0d1d_1440w.jpg" alt="img" style="zoom: 33%;" />图6

接下来以机器翻译为例，看看如何通过Seq2Seq结构把中文“早上好”翻译成英文“Good morning”：

1. 将“早上好”通过Encoder编码，并将最后 ![[公式]](https://www.zhihu.com/equation?tex=t%3D3) 时刻的隐藏层状态 ![[公式]](https://www.zhihu.com/equation?tex=h_3) 作为语义向量。
2. 以语义向量为Decoder的 ![[公式]](https://www.zhihu.com/equation?tex=h_0) 状态，同时在 ![[公式]](https://www.zhihu.com/equation?tex=t%3D1) 时刻输入<start>特殊标识符，开始解码。之后不断的将前一时刻输出作为下一时刻输入进行解码，直接输出<stop>特殊标识符结束。

当然，上述过程只是Seq2Seq结构的一种经典实现方式。**与经典RNN结构不同的是，Seq2Seq结构不再要求输入和输出序列有相同的时间长度！**

<img src="https://pic1.zhimg.com/80/v2-893e331af6b07789bbd7095c16421f2f_1440w.jpg" alt="img" style="zoom: 33%;" />



## Attention注意力机制

<img src="https://pic1.zhimg.com/80/v2-899b2e4b693704f41238893cc37f100e_1440w.jpg" alt="img" style="zoom:33%;" />

图11

在Seq2Seq结构中，encoder把所有的输入序列都编码成一个统一的语义向量Context，然后再由Decoder解码。由于context包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。如机器翻译问题，当要翻译的句子较长时，一个Context可能存不下那么多信息，就会造成精度的下降。除此之外，如果按照上述方式实现，只用到了编码器的最后一个隐藏层状态，信息利用率低下。

<img src="https://tva1.sinaimg.cn/large/007S8ZIlly1gi6qceuc7tj30u01a9dxb.jpg" alt="notes" style="zoom: 45%;" />

