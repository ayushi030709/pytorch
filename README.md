# pytorch
1. The Building Block: The Tensor
In Python, you use NumPy arrays. In PyTorch, we use Tensors.
A tensor is a multi-dimensional matrix containing elements of a single data type. Think of it as a NumPy array, but with a superpower: it can run on a GPU to accelerate computations by 100x+.

Essential Tensor Operations
Here is how we initialize and interact with data in PyTorch:

Python
import torch
import numpy as np

# 1. Creating Tensors
scalar = torch.tensor(7)               # 0-dimensional
vector = torch.tensor([7, 7])          # 1-dimensional
matrix = torch.tensor([[7, 8], [9, 10]]) # 2-dimensional (2x2)

# 2. Creating Tensors with specific initializations
zeros = torch.zeros(size=(3, 3))       # 3x3 matrix of 0s
randoms = torch.rand(size=(2, 3))     # 2x3 matrix of random numbers between 0 and 1

# 3. Moving to GPU (The speed boost)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

randoms = randoms.to(device) # Moves the tensor to GPU memory if available
2. The Engine: Autograd (Automatic Differentiation)
Neural networks learn by calculating gradients (derivatives) to minimize errors. PyTorch handles this automatically using a engine called Autograd.

When you create a tensor and set requires_grad=True, PyTorch tracks every mathematical operation performed on it. When you call .backward() on the final result, PyTorch automatically calculates the derivatives for you.

Python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 5  # Forward pass: y = 3^2 + 5 = 14

y.backward()     # Backward pass: Calculates dy/dx (which analytically is 2x)

print(x.grad)    # Output: tensor(6.0)  --> because 2 * (3.0) = 6
3. The Blueprint: How PyTorch Code is Structured
Every modern PyTorch model follows a strict, highly organized object-oriented blueprint. You create a class that inherits from torch.nn.Module.

There are always two essential functions you must define:

__init__(): Where you define the layers (the architecture/components).

forward(): Where you define the flow of data through those layers (the network's logic).

Here is a breakdown of a complete, professional neural network script:

Python
import torch
import torch.nn as nn          # nn contains all neural network layers
import torch.optim as optim    # optim contains optimization algorithms like SGD/Adam

# Step 1: Define the Network Architecture
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear represents a standard linear layer (y = xW^T + b)
        # in_features=1 (takes 1 input variable), out_features=1 (outputs 1 prediction)
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defines the computation performed at every call
        return self.linear_layer(x)

# Step 2: Instantiate the Model
model = LinearRegressionModel()
4. The PyTorch Training Loop
Once the model is built, we have to train it. A professional PyTorch training loop consists of 5 distinct, repetitive steps inside a loop.

Before starting, we must define a Loss Function (how wrong our model is) and an Optimizer (how the model adjusts its weights based on the loss).

Python
# Dummy Data: predicting y = 2x
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Define Loss and Optimizer
loss_fn = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent with Learning Rate 0.01

# The Training Loop
epochs = 100
for epoch in range(epochs):
    model.train() # Set the model to training mode
    
    # 1. Forward Pass: Compute predicted outputs by passing inputs to the model
    y_pred = model(X)
    
    # 2. Calculate Loss: Compare predictions to actual labels
    loss = loss_fn(y_pred, y)
    
    # 3. Optimizer Zero Grad: Clear accumulated gradients from the previous step
    optimizer.zero_grad()
    
    # 4. Backward Pass: Compute gradients of the loss w.r.t model parameters
    loss.backward()
    
    # 5. Optimizer Step: Update the model's weights based on the gradients
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
Explaining the Critical Commands:
optimizer.zero_grad(): Crucial. By default, PyTorch accumulates (adds up) gradients on subsequent backward passes. If you don't reset them to zero before calculating new gradients, your model's updates will be mathematically broken.

loss.backward(): Triggers the Autograd engine to calculate the gradient of the loss for every parameter.

optimizer.step(): Looks at the gradients just calculated and nudges the weights in the direction that reduces the error.

5. Evaluation Mode (Testing)
When you are ready to test your model on unseen data, you must turn off the training machinery to save memory and ensure consistent behavior.

Python
model.eval() # Turns off layers like Dropout or BatchNorm (not used here, but best practice)

with torch.inference_mode(): # Turns off gradient tracking completely (saves massive memory)
    test_input = torch.tensor([[5.0]])
    prediction = model(test_input)
    print(f"Prediction for 5.0: {prediction.item():.4f}") # Should be very close to 10.0
Summary Checklist for PyTorch Development
To write PyTorch like a professional, keep this mental map ready:

Prepare Data: Convert data into PyTorch Tensors (torch.tensor()).

Build Model: Create a subclass of nn.Module, define layers in __init__, define data flow in forward().

Choose Metrics: Select a Loss Function (nn) and an Optimizer (torch.optim).

Train: Loop through model(), loss_fn(), optimizer.zero_grad(), loss.backward(), and optimizer.step().

Evaluate: Wrap test code in model.eval() and with torch.inference_mode().
