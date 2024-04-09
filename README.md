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

