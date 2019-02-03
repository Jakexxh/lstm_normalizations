# LSTM NORMALIZATIONS

Xiaohe Xue

**该实验是基于LSTM的Normalization算法的比较实验**

**分别有 Cosin, Weight, Layer, Batch Normalization**

_本实验中代码格式和注释还有待改善和重构，望见谅_

## 1 Form of Normalizations

本实验提出了3种Normalization嵌入LSTM的方式。

3 kinds of mode are proposed in my experiment for normalizations to be embedded in LSTM.


### 1.1 normal_cells

**base, cosine normalization, weight normalization:**

![](https://ws3.sinaimg.cn/large/006tKfTcly1frk0fbnbk3j30c104ijrr.jpg)

**batch normalization, layer normalization:**

![](https://ws3.sinaimg.cn/large/006tKfTcly1frk0gweuybj30dv02j3yp.jpg)



### 1.2 normal_cells_conb

**base, cosine normalization, weight normalization:**

![](https://ws2.sinaimg.cn/large/006tKfTcly1frk0jd79d3j30bi04e74k.jpg)

**batch normalization, layer normalization:**

![](https://ws4.sinaimg.cn/large/006tKfTcly1frk0jji40uj30d002s74h.jpg)

### 1.3 normal_cells_separate

**base, cosine normalization, weight normalization:**

![](https://ws3.sinaimg.cn/large/006tKfTcly1frk0hr8dl9j30rb02bglz.jpg)

**batch normalization, layer normalization:**

![](https://ws4.sinaimg.cn/large/006tKfTcly1frk0i4ofsij30l205fq3q.jpg)



## 2 Models

### General Arguments

```bash
rnn_mode/cell/model  # cells
                     # base: base model
                     # bn_sep: batch normalization
                     # cn_sep: cosine normalization
                     # ln_sep: layer normalization
                     # wn_sep: weight noramlization
                     # pcc_sep: pcc normalization
          
lr 		  #Learning rate
g         #grain, the initial_scale of weight of noramlization
```
经多次实验验证，由于规范化算法会让神经网络对learning rate敏感度降低，在每个模型中不同规范化算法使用的最优learning rate 都是趋于一个很小的阈值，所以在最终的实验比较中，每个模型中不同规范化算法使用的learning rate 是统一的。

不同变量初始化的方法会使神经网络的效果变化，经过多次实验验证使用用Identity 和Orthogonal初始化方法等，得到不同规范化算法之间的平行差异和效果并没有由此变化，所以最终三个实验报告中所有规范化算法中权重变量的初始化方法都是统一的，最终选择的是TensorFlow中的默认生成方式——Xavier Normal initializer。 

Normalization 中scale 变量的initializer全部为*truncated_normal_initializer*。

### 1. Sequential MNIST

#### 1.1 [Data - MNIST](http://yann.lecun.com/exdb/mnist/)

#### 1.2 Introduction

神经网络的的输入是将图片转换成1*784的向量，每个time step是一个像素块。实际上，这种向量化和输入方式大大的增加了模型训练的难度，因为每次输入的值内容过于简单，循环神经网络在训练过程中很难收集可依赖的信息，正是因为这种困难延长的神经网络训练的时间，在做规范化对比试验时能更好观察分析它们的训练情况。 

LSTM中每批数据为128个；梯度下降选择Adam算法，初始learning rate 为0.01；LSTM的cell 和hidden state 大小为128，初始化方式为标准差为0.1的随机Norm分布；LSTM中权重变量按默认方式生成，其余权重变量选择随机Orthogonal方式；每单次试验，训练模型15000次即结束；所有规范化初始scale值选择1.0；当计算神经网络的中每个输出的损失时，该实验选择交叉熵。在所列的结果中，所有Base模型的训练结果都很差，这是因为其训练次数太少，达到理想的情况至少需要训练20000次，由于单次训练成本太高，而且15000次内足以看出所有规范化训练的情况，所以没有选择继续训练。 

- 在转换成784\*1的向量时，可以尝试固定的随机置换索引方式转换即permuted MNIST。 本项目还未进行尝试。
- 参考，prototype
  - https://github.com/OlavHN/bnlstm (有瑕疵)
  - https://gist.github.com/spitis/27ab7d2a30bbaf5ef431b4a02194ac60


#### 1.3 Run

```bash
python test_784*1.py --cell=base --log_dir=./logs/ --g=0.5 --lr=0.001
```

#### 1.4 Results

##### #normal_cells

| Rank | Normal               | Acc         | Loss        |
| ---- | -------------------- | ----------- | ----------- |
| 1    | Weight Normalization | 0.98165     | 0.06123     |
| 2    | Cosine Normalization | **0.97978** | **0.06319** |
| 3    | PCC Normalization    | **0.97915** | **0.06218** |
| 4    | Layer Normalization  | 0.96226     | 0.10989     |
| 5    | Batch Normalization  | 0.18503     | 2.28426     |
| 6    | Base                 | 0.10071     | 2.34336     |

##### #normal_cells_separate

| Rank | Normal               | Acc         | Loss        |
| ---- | -------------------- | ----------- | ----------- |
| 1    | Weight Normalization | 0.98200     | 0.05438     |
| 2    | Cosine Normalization | **0.98089** | **0.06101** |
| 3    | PCC Normalization    | **0.97553** | **0.07571** |
| 4    | Base                 | 0.10250     | 2.31774     |
| 5    | Batch Normalization  | 0.09789     | 2.32820     |
| 6    | Layer Normalization  | 0.09736     | 2.30080     |


##### #normal_cells_conb

| Rank | Normal               | Acc         | Loss        |
| ---- | -------------------- | ----------- | ----------- |
| 1    | Layer Normalization  | 0.97235     | 0.09181     |
| 2    | Cosine Normalization | **0.87246** | **0.34002** |
| 3    | PCC Normalization    | **0.11350** | **2.29953** |
| 4    | Weight Normalization | 0.10264     | 2.30098     |
| 5    | Base                 | 0.10189     | 2.33089     |
| 6    | Batch Normalization  | 0.10189     | 2.34810     |



### 2. PTB

#### 2.1 [Data - PTB](http://www.fit.vutbr.cz/~imikolov/rnnlm/)

Download: [simple-examples](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

#### 2.2 Introduction

Prototype: [RNN TensorFlow](https://www.tensorflow.org/tutorials/recurrent)

LSTM中每批数据为20个，使用截断反向传播，每个截断长20个time step；梯度下降方式为保持1.0的learning rate 直到第4次Epoch，之后每次Epoch 减半learning rate的数值，同时对变量的梯度具体见代码所示；LSTM的cell 和hidden state 大小为200，全部初始化为零，LSTM中权重变量按默认方式生成，其余变量按照上界0.1，下界-0.1的随机Norm分布生成；所有规范化初始scale值选择1.0。 

#### 2.3 Run

```bash
python ptb_word_lm.py --lr=1.0 --g=5.0 --rnn_mode=cn_sep --num_gpus=1 \
--save_path=./log/ptb_cob
```



#### 2.4 Results - small model

##### #normal_cells

| Normalization        |   Train | Valid   | Test        |
| -------------------- | ------: | ------- | ----------- |
| PCC Normalization    |  43.269 | 115.349 | **108.647** |
| Cosine Normalization |  43.789 | 114.863 | **108.710** |
| Weight Normalization |  45.193 | 118.892 | 113.786     |
| Base                 |  47.946 | 121.160 | 116.031     |
| Batch Normalization  |  41.104 | 132.828 | 126.505     |
| Layer Normalization  | 110.360 | 160.186 | 149.163     |

##### #normal_cells_separate

| Normalization        | Train  | Valid   | Test        |
| -------------------- | ------ | ------- | ----------- |
| PCC Normalization    | 42.738 | 114.488 | **108.398** |
| Cosine Normalization | 43.532 | 113.341 | **107.973** |
| Base                 | 41.610 | 120.849 | 114.936     |
| Weight Normalization | 43.915 | 121.311 | 115.206     |
| Batch Normalization  | 56.819 | 129.550 | 124.410     |
| Layer Normalization  | 60.453 | 134.189 | 126.715     |

##### #normal_cells_conb

| Normalization        | Train  | Valid   | Test        |
| -------------------- | ------ | ------- | ----------- |
| Cosine Normalization | 50.236 | 110.437 | **104.357** |
| PCC Normalization    | 50.209 | 110.561 | **104.431** |
| Weight Normalization | 45.983 | 121.774 | 116.431     |
| Base                 | 45.755 | 121.856 | 116.750     |
| Layer Normalization  | 60.450 | 129.629 | 122.976     |
| Batch Normalization  | 56.819 | 129.550 | 124.410     |

### 3. DRAW

#### 3.1 [Data - MNIST](http://yann.lecun.com/exdb/mnist/)

#### 3.2 Introduction

Prototype:

[char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow)

[char-rnn](https://github.com/sherjilozair/char-rnn-tensorflow)

[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

#### 3.3 Run

```bash
python train.py --model=base --lr=0.001  \
--save_dir=/tmp/char_seq100_refactor/save/base_0.001 \
--log_dir=/tmp/char_seq100_refactor/log/base_0.001
```

#### 3.4 Results

##### #normal_cells

| Rank | Normal | Scale | lr         | cost          |
| ---- | ------ | ----- | ---------- | ------------- |
| 1    | ln     | 1.0   | 0.001      | 107.849250519 |
| 2    | wn     | 1.0   | 0.001      | 107.638289787 |
| 3    | bn     | 1.0   | 0.001      | 111.273218102 |
| 4    | pcc    | 5.0   | 0.001      | 112.98063578  |
| 5    | base   | 0.0   | 0.01/0.001 | 120           |
| 6    | cn     | 5.0   | 0.001      | 123.905787323 |

##### #normal_cells_separate

| Rank | Normal | Scale | lr    | cost          |
| ---- | ------ | ----- | ----- | ------------- |
| 1    | ln     | 1.0   | 0.001 | 104.878376785 |
| 2    | cn     | 5.0   | 0.001 | 109.033839409 |
| 3    | wn     | 1.0   | 0.001 | 111.994892326 |
| 4    | pcc    | 5.0   | 0.001 | 116.779598183 |

### 4. NMT

#### 4.1 [Data - Neural Machine Translation](https://nlp.stanford.edu/projects/nmt/)

- IWSLT'15 English-Vietnamese data **[Small]**
- WMT'14 English-German data **[Medium]**


#### 4.2 Introduction
Prototype:

[Neural Machine Translation (seq2seq) Tutorial](https://github.com/tensorflow/nmt)

#### 4.3 Run

##### Small Data:

data: 133K examples, vocab=vocab.(vi|en), train=train.(vi|en) dev=tst2012.(vi|en), test=tst2013.(vi|en), [download script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/download_iwslt15.sh).

双层LSTM, batch size是128，每次实验训练12000次神经网络；梯度下降方式选择SGD，learning rate为1.0；LSTM内cell和hidden state大小为128；dropout率为0.8；余弦规范化初始scale值选择5.0其余为1.0 

```bash
python -m nmt.nmt \
    --unit_type=base \
    --src=vi --tgt=en \
    --learning_rate=1.0 \
    --grain=0.0 \
    --vocab_prefix=../../data/nmt_data/vocab  \
    --train_prefix=../../data/nmt_data/train \
    --dev_prefix=../../data/nmt_data/tst2012  \
    --test_prefix=../../data/nmt_data/tst2013 \
    --out_dir=/tmp/nmt_model/base \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```



##### Medium Data:

data: 4.5M examples, vocab=vocab.bpe.32000.(de|en),train=train.tok.clean.bpe.32000.(de|en), dev=newstest2013.tok.bpe.32000.(de|en),test=newstest2015.tok.bpe.32000.(de|en),[download script](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh)

本模型增加神经网络网络层数为四层，将Encoder变为双向循环神经网络，本模型使用的是Scaled Luong Attention[32]。batch size是1024，每次实验训练350K次神经网络；梯度下降方式选择SGD，learning rate为1.0，但是在训练170K次后没17K次learning rate 减半；LSTM内cell和hidden state大小为1024；不进行dropout；余弦规范化初始scale值选择5.0其余为1.0。 

```bash
python -m nmt.nmt \
    --unit_type=base \
    --encoder_type=bi \
    --attention=scaled_luong \
    --src=de --tgt=en \
    --vocab_prefix=../../data/nmt_data_large/wmt16_de_en/vocab.bpe.32000  \
    --train_prefix=../../data/nmt_data_large/wmt16_de_en/train.tok.clean.bpe.32000 \
    --dev_prefix=../../data/nmt_data_large/wmt16_de_en/newstest2013.tok.bpe.32000  \
    --test_prefix=../../data/nmt_data_large/wmt16_de_en/newstest2015.tok.bpe.32000 \
    --out_dir=$HOME/log/nmt_attention_model_large\
    --learning_rate=1.0 \
    --grain=1.0 \
    --start_decay_step=170000 \
    --decay_steps=17000 \
    --decay_factor=0.5 \
    --num_train_steps=350000 \
    --steps_per_stats=100 \
    --num_layers=4 \
    --num_units=1024 \
    --dropout=0.0 \
    --forget_bias=0.0 \
    --metrics=bleu
```



#### 4.4 Result

##### 4.4.1 Small Data:

##### #normal_cells

| Rank | Normal | Scale | lr   | bleu test         |
| ---- | ------ | ----- | ---- | ----------------- |
| 1    | cn     | 5.0   | 1.0  | 6.115/5.719       |
| 2    | bn     | 1.0   | 1.0  | 6.0               |
| 3    | pcc    | 5.0   | 1.0  | 5.838             |
| 4    | ln     | 1.0   | 1.0  | 5.486             |
| 5    | wn     | 1.0   | 1.0  | 5.272             |
| 6    | base   | 0.0   | 1.0  | 4.625/5.177/4.898 |

##### #normal_cells_separate

| Rank     | Normal | Scale | lr   | bleu test |
| -------- | ------ | ----- | ---- | --------- |
| 1        | cn     | 5.0   | 1.0  | 5.869     |
| 2        | pcc    | 5.0   | 1.0  | 5.819     |
| 3        | wn     | 1.0   | 1.0  | 5.559     |
| 没有效果 | ln     | /     | /    | 0.7       |



##### 4.4.1 Medium Data:

##### #normal_cells_separate

| Rank | Normal | Scale | lr   | bleu test |
| ---- | ------ | ----- | ---- | --------- |
| 1    | wn     | 1.0   | 1.0  | 30.7      |
| 2    | pcc    | 5.0   | 1.0  | 30.5      |
| 3    | cn     | 5.0   | 1.0  | 30.3      |
| 4    | ln     | 1.0   | 1.0  | 30.2      |
| 5    | base   | 0.0   | 1.0  | 27.6      |
| 6    | bn     | 1.0   | 1.0  | 崩溃      |

##### 

## Other

1. Cosine Normalization 在vanilla LSTM上没有太大效果

2. Batch Normalization:


- 因为在Batch Normalization在测试时需要用到 population mean 和 population var，所以在Batch Normalization初始化state时，除了 h 和 c ，还有step来标记当前的time step。具体请看代码。
- 因为Batch Normalization的规范化算法是纵向的，所以理论上它在normal_cells上的效果应该和在其余两个cells的效果一样，所以就没有进行试验。

