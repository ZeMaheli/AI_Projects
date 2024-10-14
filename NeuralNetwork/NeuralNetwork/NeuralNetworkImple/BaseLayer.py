class BaseLayer():
    def propagate(self, x):
        raise NotImplementedError

    def adapt(self, delta_n, y_prev, alpha, beta):
        raise NotImplementedError

    @property
    def y(self):
        raise NotImplementedError
