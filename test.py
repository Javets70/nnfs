# test.py
from tinytensor.core import Tensor

# Test 1: A simple 2D tensor
data_2d = [[1, 2], [3, 4], [5, 6]]
t1 = Tensor(data_2d)
print(f"Shape of t1: {t1.shape}")  # Should be (3, 2)
print(f"Data of t1 (flat): {t1.data}")  # Should be [1, 2, 3, 4, 5, 6]

# Test 2: A 1D tensor
data_1d = [1, 2, 3, 4]
t2 = Tensor(data_1d)
print(f"Shape of t2: {t2.shape}")  # Should be (4,)

# Test 3: A scalar
t3 = Tensor(5)
print(
    f"Shape of t3: {t3.shape}"
)  # Should be () for a 0-dimensional tensor? Let's think. For now, maybe (1,) or just handle as a list of one number.

# Test 4: Inconsistent shape (should raise an error)
# data_bad = [[1, 2], [3]]
# t4 = Tensor(data_bad) # This should raise ValueError
