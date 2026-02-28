class Tensor:
    def __init__(self, data):
        if not isinstance(data, list):
            raise ValueError("Data must be of type list")
        if len(data) == 0:
            raise ValueError("Data must be of at least length 1")
        self._shape = self._infer_shape(data)

        self._data = []
        self._flatten(data)

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

    @property
    def data(self) -> list:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._shape

    def __repr__(self):
        return f"Tensor(shape={self._shape}, data={self._data[:5]}{'...' if len(self._data) > 5 else ''})"

    def __str__(self):
        return f"Tensor({self._data}, shape={self._shape})"
