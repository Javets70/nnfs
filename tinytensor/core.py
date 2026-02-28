import random


class Tensor:
    def __init__(self, data, dtype=float):
        if not isinstance(data, list):
            raise ValueError("Data must be of type list")
        if len(data) == 0:
            raise ValueError("Data must be of at least length 1")
        self._shape = self._infer_shape(data)

        self._data = []
        self._flatten(data)

        self._dtype = dtype
        self._data = [self._dtype(i) for i in self._data]

    def _infer_shape(self, data):
        if not isinstance(data, list):
            return ()
        if not data:
            raise ValueError("Empty sublists are not allowed")

        first_shape = self._infer_shape(data[0])
        for item in data[1:]:
            if self._infer_shape(item) != first_shape:
                raise ValueError(
                    "Inconsistent nested structure: all dimensions must be rectangular"
                )

        return (len(data),) + first_shape

    def _flatten(self, data) -> list:
        if isinstance(data, list):
            for item in data:
                self._flatten(item)
        else:
            self._data.append(data)

    @classmethod
    def zeros(cls, *shape):
        """Create a tensor of given shape filled with zeros."""
        total_zeroes = 1
        for i in shape:
            total_zeroes *= i
        return cls([0.0] * total_zeroes, dtype=float)

    @classmethod
    def ones(cls, *shape):
        """Create a tensor of given shape filled with ones."""
        total_ones = 1
        for i in shape:
            total_ones *= i
        return cls([1.0] * total_ones, dtype=float)

    @classmethod
    def randn(cls, *shape):
        """Create a tensor of given shape filled with random numbers from standard normal distribution."""
        total_items = 1
        for i in shape:
            total_items *= i
        return cls([random.gauss(0, 1) for i in range(total_items)], dtype=float)

    @property
    def data(self) -> list:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self._data[:5]}{'...' if len(self._data) > 5 else ''},\
        dtype={self.dtype})"

    def __str__(self):
        return f"Tensor({self._data}, shape={self._shape}, dtype={self._dtype})"
