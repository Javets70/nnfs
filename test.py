from tinytensor.core import Tensor

# Valid cases
t1 = Tensor([[1, 2], [3, 4], [5, 6]])
print(t1)  # Tensor([1, 2, 3, 4, 5, 6], shape=(3, 2))
print(t1.shape)  # (3, 2)

t2 = Tensor([1, 2, 3, 4])
print(t2)  # Tensor([1, 2, 3, 4], shape=(4,))

t3 = Tensor([[[1, 2]], [[3, 4]]])  # shape (2, 1, 2)
print(t3.shape)  # (2, 1, 2)

# Invalid case (should raise ValueError)
try:
    t4 = Tensor([[[1, 2]], [[3]]])
except ValueError as e:
    print("Caught expected error:", e)

t_zeros = Tensor.zeros(2, 3)
print(t_zeros)  # Should show a tensor with shape (2,3) filled with 0.0

t_ones = Tensor.ones(3, 2, 4)
print(t_ones.shape)  # (3,2,4)

t_rand = Tensor.randn(5, 5)
print(t_rand)  # Should show random numbers
