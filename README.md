Aim of this project is to learn about python and create a neural network from scratch , without the
use of any external libraries.
Performance is of no concern as this is purely for learning purposes.

Example:
```python
# This is my wishful thinking code
import tinytensor as tt

# Create some data
data = tt.tensor([[1, 2], [3, 4], [5, 6]], dtype=tt.float32)
labels = tt.tensor([[0], [1], [0]], dtype=tt.float32)

# Define a simple model
class MyFirstNN:
    def __init__(self):
        self.W1 = tt.randn(2, 4)  # random weights
        self.b1 = tt.zeros(4)
        self.W2 = tt.randn(4, 1)
        self.b2 = tt.zeros(1)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1   # matrix multiply and add
        a1 = tt.sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        return tt.sigmoid(z2)

model = MyFirstNN()

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model.forward(data)
    loss = tt.binary_cross_entropy(pred, labels)

    # Backward pass (automatic differentiation)
    loss.backward()

    # Update parameters (with gradient descent)
    model.W1 -= 0.1 * model.W1.grad
    model.b1 -= 0.1 * model.b1.grad
    # ... etc
```
