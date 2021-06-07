from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger():
    def __init__(self, path):
        assert path != None, "path is None"
        self.writer = SummaryWriter(log_dir=path)

    def update_loss(self, phase, value, step):
        self.writer.add_scalar(f'{phase}/loss', value, step)

    def update_metric(self, phase, metric, value, step):
        self.writer.add_scalar(f'{phase}/{metric}', value, step)
