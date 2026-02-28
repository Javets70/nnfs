class Tensor:
    def __init__(self, data):
        if not isinstance(data, list):
            raise ValueError("Data must be of type list")
        self._data = []
        if len(data) == 0:
            raise ValueError("Data must be of atleast length 1")
        # calculating the shape
        self._shape = self._get_shape(data)

    def _get_shape(self, data) -> list:
        if isinstance(data, list):
            return [len(data)] + self._get_shape(data[0])
        return []

    def _get_data(self, data) -> list:
        if isinstance(data, list):
            if len({len(item) for item in data}) > 1:
                raise ValueError("All dimensions must be consistent")
            for item in data:
                self._get_data(item)
        else:
            for i in data:
                self._data.append(i)

    @property
    def data(self) -> list:
        return self._data

    @property
    def shape(self) -> tuple:
        return self._shape
