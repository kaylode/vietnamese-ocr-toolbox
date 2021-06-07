import torch
import torchvision.transforms as tvtf


class RandomCrop:
    def __init__(self, crop_size):
        assert isinstance(crop_size, (int, float, list, tuple)), \
            f'Invalid type, expect (int, float, list, tuple), get {type(crop_size)}'
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        if isinstance(crop_size, float):
            assert 0 < crop_size <= 1, \
                f'Invalid crop size, float should be in (0, 1], get {crop_size}'
            crop_size = (crop_size, crop_size)
        if isinstance(crop_size, (list, tuple)):
            crop_size = crop_size
        self.crop_size = crop_size

    def __call__(self, x):
        # x => C, H, W
        h, w = x.size()[-2:]

        crop_size = (min(self.crop_size[0], h),
                     min(self.crop_size[1], w))
        r0 = torch.randint(high=h - crop_size[0] + 1, size=(1,))
        c0 = torch.randint(high=w - crop_size[1] + 1, size=(1,))
        r1 = r0 + crop_size[0]
        c1 = c0 + crop_size[1]

        return r0, r1, c0, c1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.tensor(range(10000)).view(1, 1, 20, 500)

    for _ in range(10):
        r0, r1, c0, c1 = RandomCrop((150, 70))(x)

        plt.subplot(1, 2, 1)
        plt.imshow(x.squeeze(), vmin=0, vmax=10000)
        plt.subplot(1, 2, 2)
        plt.imshow(x[..., r0:r1, c0:c1].squeeze(), vmin=0, vmax=10000)
        plt.show()
        plt.close()

