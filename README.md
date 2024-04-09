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
$x_{\text{i}}$, $y_{\text{i}_{i}$, the update rule for a parameter in the model could be expressed as:
