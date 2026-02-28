# Module 1: The Core Abstraction - The Tensor
We begin at the very foundation: the Tensor object itself. In your wishful code, you use tt.tensor([[1, 2], [3, 4], [5, 6]], dtype=tt.float32). This single line is deceptively complex. It must take nested Python lists and transform them into a structured object that knows its data, its shape, and its data type.

## Lecture 1.1: What is a Tensor?
Conceptually, a tensor is a container for numbers, arranged in a regular, multi-dimensional grid. The rank of a tensor is the number of dimensions:

- Rank 0: A scalar (a single number).
- Rank 1: A vector (a 1D list, e.g., [1, 2, 3]).
- Rank 2: A matrix (a 2D grid, like your data which is 3x2).
- Rank 3 and higher: A cube or hypercube of numbers.

The shape is a tuple describing the size of each dimension. For data = [[1, 2], [3, 4], [5, 6]], the shape is (3, 2). For the labels [[0], [1], [0]], the shape is (3, 1).

Internally, how should we store this data? The most straightforward way for a beginner is to store it exactly as given: a list of lists. However, this becomes inefficient and complex for higher dimensions and operations like reshaping. The professional approach, and the one that will serve us best as we build towards performance, is to store the data in a contiguous, flat list (a 1D array) and use the shape to interpret it.

Think of it like a spreadsheet. The data is just a long list of cells. The shape tells you how many rows and columns to break it into. This is the key to many operations.

## The Design Socratic:

Let's open your core.py file. We'll start by defining our Tensor class.
```python
# tinytensor/core.py

class Tensor:
    def __init__(self, data, dtype=None):
        # We'll fill this in step-by-step
        pass
```
**What should the __init__ method do when it receives data like [[1, 2], [3, 4]]?**

1. It must determine the shape. How would you algorithmically find the shape of arbitrarily nested lists? What if the nested lists have inconsistent lengths (e.g., [[1, 2], [3]])? Should we allow that?
2. It must flatten the nested lists into a single list of numbers. How would you recursively traverse the nested structure to extract every number in the correct order?
3. It must handle the dtype. For now, let's keep it simple: we'll just try to convert every number to a standard Python float. We can worry about integers and other types later.

This single function, __init__, is where you will conquer two of your stated challenges: working with recursion (to flatten the data) and exception handling (to catch inconsistent list shapes).

### Exercise 1.1: The Seed of a Tensor
Your task is to implement the __init__ method to handle the core cases. Do not worry about performance or advanced features yet. Focus on correctness.

1. In core.py, write the __init__ method so that it:
  - Accepts data (which could be a number, a list, or a nested list).
  - Calculates the shape and stores it as self.shape.
  - Flattens all the numbers into a single Python list called self.data.
  - If the input lists are not rectangular (e.g., [[1, 2], [3]]), it should raise ValueError("All dimensions must be consistent").
  - If the dtype argument is provided, it should convert all numbers to that type (for now, just handle float32 by converting to Python float). You can ignore the dtype for now if it's easier, or just always store as float.

2. Test your code manually by creating a new file test.py in your root directory and running it. Here's a template to get you started:

```python
# test.py
from tinytensor.core import Tensor

# Test 1: A simple 2D tensor
data_2d = [[1, 2], [3, 4], [5, 6]]
t1 = Tensor(data_2d)
print(f"Shape of t1: {t1.shape}")  # Should be (3, 2)
print(f"Data of t1 (flat): {t1.data}") # Should be [1, 2, 3, 4, 5, 6]

# Test 2: A 1D tensor
data_1d = [1, 2, 3, 4]
t2 = Tensor(data_1d)
print(f"Shape of t2: {t2.shape}") # Should be (4,)

# Test 3: A scalar
t3 = Tensor(5)
print(f"Shape of t3: {t3.shape}") # Should be () for a 0-dimensional tensor? Let's think. For now, maybe (1,) or just handle as a list of one number.

# Test 4: Inconsistent shape (should raise an error)
# data_bad = [[1, 2], [3]]
# t4 = Tensor(data_bad) # This should raise ValueError
```

## Lecture 1.2: Adding Data Type and Factory Functions
Your tensor currently stores all numbers as Python objects (their original type). For numerical computing, we typically want to control the data type (e.g., all floats) for consistency and to prepare for future optimizations. Also, your __init__ requires a list, but we often want to create tensors filled with zeros or random numbers directly.

### Step 1: Adding dtype support
Modify your __init__ to accept an optional dtype parameter. For now, let's support two types: int and float. If dtype is provided, convert every element in self._data to that type during flattening. If not provided, we can infer the type from the first element (or default to float for safety).

**Socratic Question:**
Where should the conversion happen? Inside _flatten, or after flattening? What are the trade-offs?

Hint: If you convert during flattening, you avoid an extra loop, but you need to pass the dtype down the recursion. If you convert after, you keep flattening simple and then apply a list comprehension. For clarity, let's do it after flattening.

### Exercise 1.2.1: Implement dtype
1. Add a dtype parameter to __init__ with a default value of float. Store it as self.dtype.
2. In __init__, after flattening, convert every element in self._data to self.dtype. You can use a list comprehension: self._data = [self.dtype(x) for x in self._data].
3. Update __repr__ and __str__ to show the dtype.

### Step 2: Factory Functions as Class Methods
In your wishful code, you used tt.zeros(4) and tt.randn(2, 4). These are not methods of a tensor instance; they are functions that return a new tensor. In Python, we can implement them as class methods—methods that belong to the class itself, not to an instance.
A class method receives the class as the first argument (conventionally cls). We'll use them to create tensors of a given shape.

### Exercise 1.2.2: Implement zeros, ones, and randn

Add these class methods to your Tensor class:

```python
@classmethod
def zeros(cls, *shape):
    """Create a tensor of given shape filled with zeros."""
    # Hint: shape is a tuple of integers, e.g., zeros(2, 3) means shape (2,3)
    # You need to create a flat list of zeros of length product(shape)
    # Then call cls(data, dtype=float) or whatever dtype you want.
    pass

@classmethod
def ones(cls, *shape):
    """Create a tensor of given shape filled with ones."""
    pass

@classmethod
def randn(cls, *shape):
    """Create a tensor of given shape filled with random numbers from standard normal distribution."""
    # Use random.gauss(0, 1) or random.normalvariate(0,1) from Python's random module.
    # Import random at the top of your file.
    pass
```

**Socratic Questions:**
- How do you compute the total number of elements from a shape tuple like (2, 3)? (Product of dimensions)
- For randn, you need to generate random numbers. Which Python module provides Gaussian random numbers? (random has gauss and normalvariate.)
- Should these methods accept a dtype parameter? For now, assume float.

Add tests in test.py:

```python
t_zeros = Tensor.zeros(2, 3)
print(t_zeros)  # Should show a tensor with shape (2,3) filled with 0.0

t_ones = Tensor.ones(3, 2, 4)
print(t_ones.shape)  # (3,2,4)

t_rand = Tensor.randn(5, 5)
print(t_rand)  # Should show random numbers
```
