# LSTM_NORMALIZATIONS

Xiaohe Xue



## FORM of Normalizations

### normal_cells

$$
\begin{bmatrix}
i \ f \ g \ o
\end{bmatrix}
=

norm(h_{t-1} ,W_{h})  \ +  \ norm(x_{t}, W_{x})

\\
new\_c = (c * \sigma(f + bias) + \sigma(i) * \tanh(j))
\\
(new\_c = batch\_norm(new\_c) \  or  \ layer\_norm (new\_c))
\\
new\_h = \tanh(new\_c) * \sigma(o)
$$



### normal_cells_conb

$$
\begin{bmatrix}
i \ f \ g \ o
\end{bmatrix}
=

norm(\begin{bmatrix} h_{t-1} ,\  x_{t} \end{bmatrix}, W)

\\
new\_c = (c * \sigma(f + bias) + \sigma(i) * \tanh(j))
\\
(new\_c = batch\_norm(new\_c) \  or  \ layer\_norm (new\_c))
\\
new\_h = \tanh(new\_c) * \sigma(o)
$$

### normal_cells_separate

$$
\begin{bmatrix}
i \ f \ g \ o
\end{bmatrix}
=

[norm( h_{t-1},  w_{ih}) ,\  norm( h_{t-1},  w_{fh}) ,\  norm( h_{t-1},  w_{gh}), \  norm( h_{t-1},  w_{oh})] + \ norm(x_{t}, W_{x})

\\
new\_c = (c * \sigma(f + bias) + \sigma(i) * \tanh(j))
\\
(new\_c = batch\_norm(new\_c) \  or  \ layer\_norm (new\_c))
\\
new\_h = \tanh(new\_c) * \sigma(o)
$$



## Sequential MNIST



## Other

Cosine Normalization does not well on all vanilla LSTM model