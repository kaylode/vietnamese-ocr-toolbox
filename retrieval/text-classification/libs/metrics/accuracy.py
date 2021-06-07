import torch


class Accuracy:
    def __init__(self, *args, **kwargs):
        self.reset()

    def calculate(self, output, target):
        pred = torch.argmax(output, dim=1,)
        correct = (pred == target).sum()
        sample_size = output.size(0)
        return correct, sample_size

    def update(self, value):
        self.correct += value[0]
        self.sample_size += value[1]

    def reset(self):
        self.correct = 0.0
        self.sample_size = 0.0

    def value(self):
        return self.correct / self.sample_size

    def summary(self):
        print(f"Accuracy: {self.value()}")


if __name__ == "__main__":
    metric = Accuracy()
    metric.reset()
    pred = torch.tensor([[2, 1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 1]])
    lbls = torch.tensor([0, 2, 3])
    value = metric.calculate(pred, lbls)
    metric.update(value)
    print(metric.value())
