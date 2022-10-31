import numbers
import random
import torchvision.transforms as tf
import torchvision.transforms.functional as tff

from PIL import Image, ImageFilter, ImageChops

def _assert(img, mask):
    assert img.size == mask.size, f'{img.size} vs {mask.size}'


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, img, mask):
        if isinstance(mask, list):
            for m in mask:
                if m is not None:
                    _assert(img, m)
        else:
            _assert(img, mask)
        
        kwargs = {}
        for a in self.augmentations:
            img, mask, kwargs = a(img, mask, kwargs)

        return img, mask, kwargs


class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask, kwargs):
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, mask, kwargs
        if w < tw or h < th:
            return (
                img.resize((tw, th), Image.BILINEAR),
                [m.resize((tw, th), Image.NEAREST) if m is not None else None for m in mask],
                kwargs
            )
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            [m.crop((x1, y1, x1 + tw, y1 + th)) if m is not None else None for m in mask],
            kwargs
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, kwargs):
        flip = random.random() < self.p

        kwargs['RandomHorizontallyFlip'] = flip
        if flip:
            return (
                img.transpose(Image.FLIP_LEFT_RIGHT),
                [m.transpose(Image.FLIP_LEFT_RIGHT) if m is not None else None for m in mask],
                kwargs
            )
        return img, mask, kwargs


class RandomSized(object):
    def __init__(self, rng, size):
        self.rng = rng
        self.size = size

    def __call__(self, img, mask, kwargs):
        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(*self.rng) * self.size)
        h = int(w / prop)

        img = img.resize((w, h), Image.BILINEAR)
        mask = [m.resize((w, h), Image.NEAREST) if m is not None else None for m in mask]

        return img, mask, kwargs


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.radius = [0.1, 2.0]
        self.p = p

    def __call__(self, img, mask, kwargs):
        if random.random() <= self.p:
            radius = random.uniform(self.radius[0], self.radius[1])
            return img.filter(ImageFilter.GaussianBlur(radius)), mask, kwargs
        return img, mask, kwargs


class RandomColorJitter(object):
    def __init__(self, jitter, p=0.5):
        self.p = p
        self.jitter = tf.ColorJitter(brightness=jitter,
                                     contrast=jitter,
                                     saturation=jitter,
                                     hue=min(0.1, jitter))

    def __call__(self, img, mask, kwargs):
        if random.random() <= self.p:
            img = self.jitter(img)

        return img, mask, kwargs


class RandomGreyscale(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img, mask, kwargs):
        if random.random() <= self.p:
            img = tff.to_grayscale(img, num_output_channels=3)

        return img, mask, kwargs


def image_shift(img, mask, rate_h, rate_w):
    w, h = img.size
    shift_h = int(h * rate_h)
    shift_w = int(w * rate_w)
    img = ImageChops.offset(img, shift_w, shift_h)
    mask = [ImageChops.offset(m, shift_w, shift_h) for m in mask]
    return img, mask


class RandomShift(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, kwargs):
        if random.random() < self.p:
            shift_rate_h = random.random()
            shift_rate_w = random.random()
            img, mask = image_shift(img, mask, shift_rate_h, shift_rate_w)
            kwargs['RandomShift'] = (shift_rate_h, shift_rate_w)
        else:
            kwargs['RandomShift'] = (0, 0)
        return img, mask, kwargs
