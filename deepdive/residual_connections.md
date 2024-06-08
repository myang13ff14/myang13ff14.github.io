---
layout: default
title: Residual Connections
---

# Purpose of Residual Connections

## Introduction

In deep neural networks, one common issue is the vanishing gradient problem. As the network gets deeper, the gradients of the loss function with respect to the parameters can become very small, causing slow or stalled learning. Residual connections, also known as shortcut connections, help mitigate this issue by providing an alternative path for the gradient to flow through the network. This ensures that gradients can be backpropagated more effectively, leading to better performance and faster convergence.

The `CustomDeepNetwork` class below demonstrates the implementation of residual connections. The `use_residual` parameter determines whether the shortcut connections are applied. When `use_residual` is set to True, the input x is added to the output of each layer if their shapes match, creating a residual connection. This can help the model learn more efficiently and achieve better performance.

## Implementation

Below is the implementation of K-means clustering using PyTorch:

```python
import torch
import torch.nn as nn

class CustomDeepNetwork(nn.Module):
    def __init__(self, layers, use_residual):
        super(CustomDeepNetwork, self).__init__()
        self.use_residual = use_residual
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                nn.GELU()
            ))
    
    def forward(self, x):
        for layer in self.layers:
            output = layer(x)
            if self.use_residual and x.shape == output.shape:
                x = x + output
            else:
                x = output
        return x

# Define the sizes of each layer
layer_sizes = [3,3,3,3,3,3,3,3,3,3,3,3,3,31]
# Initialize the model with residual connections
model_with_residual = CustomDeepNetwork(layer_sizes, use_residual=True)
model_without_residual = CustomDeepNetwork(layer_sizes, use_residual=False)

def display_gradients(network, input_tensor):
    # Perform a forward pass through the model
    prediction = network(input_tensor)
    target = torch.zeros_like(prediction)
    
    # Compute the mean squared error loss
    criterion = nn.MSELoss()
    loss_value = criterion(prediction, target)
    
    # Perform a backward pass to compute gradients
    loss_value.backward()
    
    # Iterate through model parameters and print gradients
    for param_name, parameter in network.named_parameters():
        if 'weight' in param_name:
            # Print the gradient values and their mean absolute value
            print(f"Mean absolute gradient of {param_name}: {parameter.grad.abs().mean().item()}")

torch.manual_seed(12345)
sample_input = torch.ones(3).clone().detach()


display_gradients(model_with_residual, sample_input)
```

``` css
Mean absolute gradient of layers.0.0.weight: 7.344178199768066
Mean absolute gradient of layers.1.0.weight: 8.662698745727539
Mean absolute gradient of layers.2.0.weight: 13.765417098999023
Mean absolute gradient of layers.3.0.weight: 7.862175941467285
Mean absolute gradient of layers.4.0.weight: 8.265609741210938
Mean absolute gradient of layers.5.0.weight: 2.184358835220337
Mean absolute gradient of layers.6.0.weight: 7.696049213409424
Mean absolute gradient of layers.7.0.weight: 7.910844802856445
Mean absolute gradient of layers.8.0.weight: 3.925398826599121
Mean absolute gradient of layers.9.0.weight: 4.45281457901001
Mean absolute gradient of layers.10.0.weight: 9.0689697265625
Mean absolute gradient of layers.11.0.weight: 5.701756000518799
Mean absolute gradient of layers.12.0.weight: 1.750020980834961
```

```python
display_gradients(model_without_residual, sample_input)
```

```css
Mean absolute gradient of layers.0.0.weight: 7.344178199768066
Mean absolute gradient of layers.1.0.weight: 8.662698745727539
Mean absolute gradient of layers.2.0.weight: 13.765417098999023
Mean absolute gradient of layers.3.0.weight: 7.862175941467285
Mean absolute gradient of layers.4.0.weight: 8.265609741210938
Mean absolute gradient of layers.5.0.weight: 2.184358835220337
Mean absolute gradient of layers.6.0.weight: 7.696049213409424
Mean absolute gradient of layers.7.0.weight: 7.910844802856445
Mean absolute gradient of layers.8.0.weight: 3.925398826599121
Mean absolute gradient of layers.9.0.weight: 4.45281457901001
Mean absolute gradient of layers.10.0.weight: 9.0689697265625
Mean absolute gradient of layers.11.0.weight: 5.701756000518799
Mean absolute gradient of layers.12.0.weight: 1.750020980834961
```

In our experiments with `CustomDeepNetwork`, we observed that the vanishing gradient problem is more pronounced in models without residual connections compared to those with residual connections. This phenomenon can be attributed to the way gradients are propagated through layers during backpropagation in standard deep networks. Without residual connections, the gradients are sequentially multiplied through many layers. When small gradients are multiplied together, they can diminish exponentially, leading to the vanishing gradient problem.

Residual connections introduce skip connections or shortcuts that enable gradients to bypass certain layers during backpropagation. This mechanism significantly mitigates the vanishing gradient problem.


Compare the differences:

$$L(x) = f(g(x))$$

$$\dfrac{dL}{dx} = df/dg \times dg/dx$$

$$L(x) = f(g(x)) + x$$

$$\dfrac{dL}{dx} = df/dg \times dg/dx + 1$$

By adding a residual link, we are actually adding 1 directly to the equation.

Interestingly, our experiments with `CustomDeepNetwork` also revealed that increasing the width of each layer can help alleviate the vanishing gradient problem to some extent.

Let's explore the following setup further:

```python
layer_sizes = [128,256,512,1024,512,256,128,1]
display_gradients(model_with_residual, sample_input)
```

```css
Mean absolute gradient of layers.0.0.weight: 4.922050607092388e-07
Mean absolute gradient of layers.1.0.weight: 2.632298787830223e-07
Mean absolute gradient of layers.2.0.weight: 2.0730938388169307e-07
Mean absolute gradient of layers.3.0.weight: 3.326572368678171e-07
Mean absolute gradient of layers.4.0.weight: 7.172093319240957e-07
Mean absolute gradient of layers.5.0.weight: 3.7833722217328614e-06
Mean absolute gradient of layers.6.0.weight: 0.00021849258337169886
```
```python
display_gradients(model_without_residual, sample_input)
```
```css
Mean absolute gradient of layers.0.0.weight: 6.812519472987333e-07
Mean absolute gradient of layers.1.0.weight: 3.5042324952883064e-07
Mean absolute gradient of layers.2.0.weight: 2.8378119054650597e-07
Mean absolute gradient of layers.3.0.weight: 4.569472196180868e-07
Mean absolute gradient of layers.4.0.weight: 9.6639780622354e-07
Mean absolute gradient of layers.5.0.weight: 6.078783826524159e-06
Mean absolute gradient of layers.6.0.weight: 0.0004158136434853077
```

It seems that the effect of the residual link has disappeared.