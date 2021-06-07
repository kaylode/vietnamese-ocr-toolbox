class MetricTemplate():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def calculate(self, output, target):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
