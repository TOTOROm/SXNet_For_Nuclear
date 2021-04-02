import numpy as np
import cv2
import random
import string


def add_text_noise(img, stddev):
    img = img.copy()
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_for_cnt = np.zeros((h, w), np.uint8)
    occupancy = np.random.uniform(0, stddev)

    while True:
        n = random.randint(5, 10)
        random_str = ''.join([random.choice(string.ascii_letters + string.digits) for i in range(n)])
        font_scale = np.random.uniform(0.5, 1)
        thickness = random.randint(1, 3)
        (fw, fh), baseline = cv2.getTextSize(random_str, font, font_scale, thickness)
        x = random.randint(0, max(0, w - 1 - fw))
        y = random.randint(fh, h - 1 - baseline)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.putText(img, random_str, (x, y), font, font_scale, color, thickness)
        cv2.putText(img_for_cnt, random_str, (x, y), font, font_scale, 255, thickness)

        if (img_for_cnt > 0).sum() > h * w * occupancy / 100:
            break
    return img


def add_gaussian_noise(img, stddev):
    noise = np.random.randn(*img.shape) * stddev
    noise_img = img + noise
    noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
    return noise_img


def add_impulse_noise(img, stddev):
    occupancy = np.random.uniform(0, stddev)
    mask = np.random.binomial(size=img.shape, n=1, p=occupancy / 100)
    noise = np.random.randint(256, size=img.shape)
    img = img * (1 - mask) + noise * mask
    return img.astype(np.uint8)


def add_multi_noise(img, stddev):
    img = img.copy()
    g = add_gaussian_noise(img, stddev)
    t = add_text_noise(g, stddev)
    i = add_impulse_noise(t, stddev)
    return i
