# Batch-Size

The batch size is a hyperparameter that defines the number of samples to work through before
updating the internal model parameters. In the context of training machine learning models,
particularly neural networks, data is often divided into batches for computational efficiency and
practicality. The choice of batch size can significantly affect the performance of the model in terms of
speed, convergence, and accuracy. Here's a detailed look at the implications and considerations
associated with batch size:

- **Batch Gradient Descent:** The entire dataset is considered a single batch. The model parameters
are updated once per epoch after computing the gradient of the loss function over the entire
dataset.
- **Stochastic Gradient Descent (SGD):** The batch size is set to one. Thus, the model parameters are
updated after computing the gradient of the loss function for every sample. This approach can
lead to very noisy updates, but it also allows for online training.
- **Mini-batch Gradient Descent:** This is a compromise between the two extremes mentioned
above. The dataset is divided into mini-batches of a specific size, and model parameters are
updated for each mini-batch. This is the most common approach used in practice.

<h2 align="center">Impact of Batch Size</h2>

- **Computational Efficiency:** Larger batch sizes allow for more efficient use of hardware
accelerators like GPUs due to better parallelization. However, they also require more memory.
- **Convergence Behavior:** Small batch sizes introduce noise into the parameter updates, which can
help escape local minima but might also lead to convergence instability. Large batch sizes
provide more stable but possibly less flexible updates.
- **Generalization:** Empirical evidence suggests that smaller batch sizes often lead to models that
generalize better to unseen data. This is thought to be due to the noise introduced by smaller
batches, which acts as a form of regularization.
- **Memory Constraints:** The maximum batch size can be limited by the memory capacity of the
hardware used for training. This often requires balancing between batch size and model
complexity.

<h2 align="center">Choosing Batch Size</h2>

The choice of batch size is usually empirical and can depend on various factors, including:

- The specific machine learning task and dataset.
- Hardware constraints, especially GPU memory.
- The desire for faster convergence versus better generalization.

The batch size $N$, the learning rate by $\alpha$, and the loss function by $L$. For a given batch of samples
$x_{\text{i}}$, $Y_{i_i}^N = 1$, the update rule for a parameter in the model could be expressed as:<br/>

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{1}{N} \ \sum_{i=1}^N \nabla L(w_{\text{old}}; x_{\text{i}}, y_{\text{i}})$$

This represents the average gradient of the loss function over the batch of samples, scaled by the
learning rate.
- Start with a small batch size, then increase until you see diminishing returns in speed and
  learning stability.
- Consider the hardware limitations and adjust the batch size to fully utilize the computational
  power without exceeding memory capacity.
- Use empirical testing and validation to find the batch size that offers the best compromise
  between training speed and model performance.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42) # for reproducibility
x = 2 * np.random.rand(100, 1) # 100 samples, single feature
y = 4 + 3 * x + np.random.randn(100, 1)

def predict(x, w, b):
    return x.dot(w) + b

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def gradients(x, y, y_pred):
    error = y_pred - y
    grad_w = 2 * x.T.dot(error) / len(x)
    grad_b = 2 * np.mean(error)
    return grad_w, grad_b
..........................

```

Here We used a dataset with a single feature for simplicity, aiming to predict a target variable through
linear regression. This model will have two parameters: weight and bias , which we'll update using
gradient descent.

The steps are as follows:

  1. Generate a synthetic dataset.
  2. Define the model and loss function.
  3. Implement mini-batch gradient descent.
  4. Train with different batch sizes and observe the effects.

After training with different batch sizes—stochastic , small , medium , and full batch —here
are the final weights and biases learned by the model for each batch size:

- **Stochastic (Batch Size = 1):** $w = 2.74$, $b = 4.19$
- **Small Batch (Batch Size = 10):** $w = 2.79$, $b = 4.20$
- **Medium Batch (Batch Size = 50):** $w = 2.67$, $b = 4.32$
- **Full Batch (Batch Size = 100):** $w = 2.94$, $b = 3.19$

This demonstrates how the batch size can influence the final parameters learned by the model. The
variations in and for different batch sizes reflect the differences in how the gradient is estimated
and applied:

- **Stochastic gradient descent** (batch size = 1) updates the parameters in a highly variable manner,
  leading to a noisy but potentially useful exploration of the parameter space.
- **Small and medium batch sizes** offer a balance, providing more stable updates than stochastic
  gradient descent but more variability than full batch learning. This can help in finding a good
  balance between exploration and exploitation of the parameter space.
- **Full batch gradient descent** uses the entire dataset to compute the gradient, resulting in the
  most stable update direction but potentially getting stuck in local minima or taking longer to
  converge due to the lack of noise in the updates.
  
The choice of batch size affects the training dynamics, convergence speed, and the ability to escape
local minima. Smaller batches may lead to faster convergence times in practice due to more frequent
updates, but they can also introduce more noise into the training process, which can be both
beneficial (for escaping local minima) and detrimental (leading to less stable convergence).

```python
plt.subplot(1, 2, 1)
plt.bar(range(len(batch_sizes)), weights, tick_label = batch_sizes)
plt.xlabel('Batch Size')
plt.ylabel('Weight value')
plt.title('Final Weight for Different Batch Sizes')

# biases 

plt.subplot(1, 2, 2)
plt.bar(range(len(batch_sizes)), weights, tick_label = batch_sizes)
plt.xlabel('Batch Size')
plt.ylabel('Biases value')
plt.title('Final Bias for Different Batch Sizes')

plt.tight_layout()
plt.show()

```
![image](https://github.com/Rifat-Ahammed/Batch-Size/assets/96107279/5354358f-563f-4acb-9743-d1d9d2184120)

- The left chart shows the **Final Weight** for different batch sizes. You can observe that the weight
 values are quite close to each other, with slight variations. This indicates that while the batch size
 does influence the learning process, the model is able to learn a similar underlying pattern across
 different batch sizes.

- The right chart displays the **Final Bias** for the same batch sizes. Similar to the weight values, the
 bias values also show variations but remain in a comparable range.
