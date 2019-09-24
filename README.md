# Neural network cheat sheet

### Basic procedures
1. Initialize parameters / Define hyperparameters
2. Forward propagation
3. Cost function
4. Backward propagation
5. Update parameters
6. Use trained parameters to predict

### Normalizing training sets
1. Subtract mean & Normalize variance to 1
2. Make sure training set and test set using the same mean and variance

### Initialize parameters
1. Weight matrix (W) & Bias vector (b)
2. For layer l , W.shape = (layer_dims[l], layer_dims[l - 1]), b.shape = (layer_dims[l], 1)
3. To avoid vanishing / exploding gradients, the scaling factor is sqrt(2/layers_dims[l-1]) (He initialization). Xavier initialization sqrt(1/layers_dims[l-1]) is more oftenly used when activation function is tanh.

```
def initialize_parameters(layer_dims):
  parameters = {}
  L = len(layer_dims) - 1
  for l in rang(1, L + 1):
    parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
    parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
  return parameters
```
### Forward propagation
1. Z(l) = np.dot(W(l), A(l-1)) + b(l)
2. A(l) = relu(Z(l))
3. Commonly used activation functions : Relu , Sigmoid, Tanh, Softmax, Leaky ReLU
4. Caches value of A W b Z for backward propagation

```
def linear_forward(A, W, b):
  Z = np.dot(W,A) + b
  return Z

def L_model_forward(X, parameters):
  A = X
  L = len(parameters)
  for l in range(1,L):
    A_prev = A
    Z = linear_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
    A = relu(Z)
  #Output layer uses 'sigmoid'
  ZL = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
  AL = sigmoid(ZL)
  return AL
```
### Cost Function
To use framework like Tensorflow, it's very easy to compute the cost.

```
tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
```

### Regularization
1. L1,L2 Regularization.
2. L1 is sum of abs(W), W vector will have lots of zeros.
3. L2 is sum of square(W)
4. L2 Regularization is used much more often.
```
sum = 0
for l in range(1,L):
    sum += np.sum(np.square('W'+str(l)))
L2_regularization_cost = lambd * sum / (2 * m)
cost = cost + L2_regularization_cost
```
5. Dropout
```
#In forward propagation
keep_prob = 0.5
A1 = relu(Z1)
D1 = np.random.rand(A1.shape[0], A1.shape[1])
D1 = D1 < keep_prob
A1 = A1 * D1
A1 = A1 / keep_prob
#In backward propagation
dA1 = dA1 * D1
dA1 = dA1 / keep_prob
```

### Backward propagation & Update parameters
To use TensorFlow for example
```
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
_ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
```

1. mini-batch gradient descent
2. Gradient descent
3. Gradient descent with momentum
4. RMSprop
5. Adam

### Batch Norm
Normalization on W. Make the model more robust, faster and some kind of Regularization.

### Tuning hyperparameters

- learning_rate - alpha

- Momentum - Beta = 0.9
- hidden units
- mini-batch size
- Batch Norm alpha beta

- number of layers
- learning rate decay
- Adam beta1=0.9, beta2=0.999, epsilon=1e-8
