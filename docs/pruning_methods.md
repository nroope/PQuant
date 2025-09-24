# Descriptions of the pruning methods

Our implementations follow the actual implementations of the author's of the papers, whenever we were able to find one. Because of this some of the functionality of the pruning methods can differ slightly from the equations shown in the papers.

#### [Activation pruning](https://arxiv.org/abs/1903.04476)
Collect layer outputs to calculate average layer activity (how often layer neuron / channel outputs values greater than 0). Prune those neurons and channels which have smaller activity value than a given threshold.

**Hyperparameters**
- `threshold`: If a neuron or channel is less active than this threshold, prune it.
- `threshold_decay`: Not used.
- `t_delta`: How many batches to collect as calibration data.
- `t_start_collecting_batch`: At which epoch during training the collection begins

#### [AutoSparse](https://arxiv.org/abs/2304.06941)
$x = sign(W) \cdot ReLU(|W| - \sigma(T))$.
```math
g = \begin{cases}
    1, & \text{if W > 0} \\
    \alpha, & \text{otherwise}\quad,
\end{cases}
```
where T is threshold, W is the weight matrix, g is the gradient.
 $\alpha$ is decayed after each epoch using cosine sigmoid decay.

**Hyperparameters:**
- `alpha`: initial value for $\alpha$
- `backward_sparsity`: if true, sets gradients to 0 for weights in the bottom 50% magnitude of weights in the layer. False in the default config.
- `threshold_decay`: threshold decay for optimizer. 0 in the default config.
- `threshold_init`: initial value for threshold. -5 in the default config.
- `threshold_type`: weightwise/channelwise/layerwise. Defines whether each weight has its own threshold, or is threshold shared between weights in a channel, or does the whole layer have one threshold.

#### [Continuous Sparsification](https://arxiv.org/abs/1912.04427)
A multi-round pruning algorithm.
```math
 x = W\cdot M
```
where
```math
M=(\frac{\sigma(\beta s)}{\sigma(s_{init})})
```
$\beta$ starts from the initial value at the beginning of each round, and increased exponentially until reaching a final value. $s$ is a learnable matrix with a same shape as the weight matrix. $s_{init}$ is the initial value of $s$.

During each round, as the $s$ matrix is learning and the $\beta$ is increased, the values of the mask get pushed more and more towards 0 and 1. After each round, $\beta$ is reset, and the positive values of $s$ are set to $s_{init}$ value, and negative values are kept as they are. This means that the weights pruned by $s$ stay pruned after each round, but the weights that have not been pruned previously can be pruned after a new round begins, since their values are reset in $s$.

Before fine-tuning the mask is fixed and converted to a hard mask of 0s and 1s, and all the weights rewinded back to an earlier state.

**Hyperparameters**
- `final_temp`: Value up to which $\beta$ is increased during each round. 200 in the default config.
- `threshold_decay`: L1 decay for the $s$ matrix. 1.0e-09 in the default config.
- `threshold_init`: Initial value for $s$. 0 in the default config. Lower value means more pruning, higher value means less pruning.


#### [DST](https://arxiv.org/abs/2005.06870)
$x = ReLU(|W| - T)$.
```math
g = \begin{cases}
    2-4\cdot|W|, & \text{if } |x| \leq 0.4 \\
    0.4, & \text{if } 0.4 < |x| \leq 1 \\
    0, & \text{if }|x| > 1\quad.
\end{cases}
```
The threshold T is controlled by additional loss, which is calculated by
```math
\alpha \cdot \sum_{i,j}{e^{-T_{i,j}}}
```

**Hyperparameters**
- `alpha`: Used to control the threshold via loss. 5.0e-06 in the default config.
- `max_pruning_pct`: The algorithm has a tendency to prune whole layers, so if pruning goes higher than this value, reset the threshold. 0.99 in the default config.
- `threshold_decay`: threshold decay for optimizer. 0 in the default config.
- `threshold_init`: Initial value for threshold. 0 in the default config.
- `threshold_type`: weightwise/channelwise/layerwise. Defines whether each weight has its own threshold, or is threshold shared between weights in a channel, or does the whole layer have one threshold.



#### [PDP](https://arxiv.org/abs/2305.11203)
Captures weight distribution of each layer and calculates a threshold, then does a softmax between the weights and this value, creating a soft mask.


$`W_h = topK(|W|, (1-r) \cdot n(W))\newline`$\
$`W_i = bottomK(|W|, r \cdot n(W))`$\
$`t = 0.5 \cdot (min(W_h) + max(W_i))`$\
$`zw, mw = softmax(\frac{t^2, w^2}{\tau})\text{ for $w$ in $W$}`$\
$`w = mw \cdot w`$,

where $\tau$ is the temperature, $r$ is the target sparsity of the layer for that iteration, $n(W)$ is the number of weights. The $mw$ in the above equation will have all the softmax values of the weights between the weight tensor and the threshold. If a weight is above the threshold, due to the temperature, the softmax result will very quickly go towards 1. The $r$ is increased linearly during training. The layerwise budget sparsity is calculated after a pre-training phase, in a way that the total sparsity of the model is the target sparsity given in the config.

PDP has an unstructured, N:M pruning (not yet implemented here), and channel pruning version.

**Hyperparameters**
  `epsilon`: How fast to increaes the sparsity during training. After each epoch, the sparsity is increased by this amount, until the value reaches 1 (100% of target sparsity). 0.015 in the default config, which means after ~70 epochs the target sparsity has been reached.
- `sparsity`: Target sparsity for the whole model
- `temperature`: Temperature of the softmax. 1e-5 in the default config
- `threshold_decay`: Not used
- `structured_pruning`: Whether to use a structured pruning variant or not. Structured pruning uses l2 norms of the channels/neurons instead of absolute values of weights when calculating the threshold, and prunes whole channels/neurons using that threshold value.

#### [Wanda](https://arxiv.org/abs/2306.11695)
One shot pruning, originally a post-training pruning method without fine-tuning (to implement the post-training version is on the to-do list).

Using a calibration data set, calculate a metric based on the average input to the layer, and multiply the absolute values of the weights with that metric. Prune weights based on this multiplication result (lowest values being pruned first), until a target sparsity has been reached.

For linear layers, the metric is calculated as L2 norm over the batch dimension. For convolutions, reduce dimensions by taking the average of the batch dimension, then calculate L2 norm over a flattened kernel dimension.

**Hyperparameters**
- `calculate_pruning_budget`: If True, calculate the pruning budget for each layer, while keeping the target sparsity. If False, prunes every layer using target sparsity.
- `M`: If doing N:M pruning, N and M should be non-null (N < M)
- `N`: If doing N:M pruning, N and M should be non-null (N < M)
- `threshold_decay`: not used
- `sparsity`: target sparsity. 0.9 in the default config
- `t_delta`: how many batches to collect as calibration data
- `t_start_collecting`: training step when collection starts
