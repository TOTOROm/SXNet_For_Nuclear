import numpy as np
import cv2
import math
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

MAX = math.pow(2, 14)


def get_high_low_by_fft(data, radius=10, vis=False):
    fre = np.fft.fft2(data)  # 变换得到的频域图数据是复数组成的
    fre_shift = np.fft.fftshift(fre)  # 把低频数据移到频域图的中央，便于处理
    rows, cols = data.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.ones((rows, cols))
    mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 0  # 把中间的低频部分去掉，所以是高通滤波
    f = fre_shift * mask
    data_high = np.abs(np.fft.ifft2(f))

    mask = np.zeros((rows, cols))
    mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1
    f = fre_shift * mask
    data_low = np.abs(np.fft.ifft2(f))
    if vis:
        plt.subplot(221)
        plt.axis('off')
        plt.title('Source')
        plt.imshow(data)
        # plt.subplot(222)
        # plt.axis('off')
        # plt.title('Down')
        # plt.imshow(data_down)
        plt.subplot(223)
        plt.axis('off')
        plt.title('Low')
        plt.imshow(data_low)
        plt.subplot(224)
        plt.axis('off')
        plt.title('High')
        plt.imshow(data_high)
        plt.show()
    return data_high, data_low


def get_high_low_by_pyramid(data, factor=40, vis=False):
    def avg_pooling(data, pooling_stride):
        h, w = data.shape
        h2, w2 = h // pooling_stride, w // pooling_stride
        new_data = []
        for i in range(0, h, pooling_stride):
            for j in range(0, w, pooling_stride):
                data_tmp = data[i:i + pooling_stride, j:j + pooling_stride]
                new_data.append(np.mean(data_tmp))
        new_data = np.array(new_data).reshape(h2, w2)
        return new_data

    data_down = avg_pooling(data, pooling_stride=factor)
    data_low = cv2.resize(data_down, (data.shape[1], data.shape[0]), interpolation=cv2.INTER_CUBIC)
    data_high = data - data_low
    print('data_high_mean:',np.mean(data_high))
    if vis:
        plt.subplot(221)
        plt.axis('off')
        plt.title('Source')
        plt.imshow(data)
        plt.subplot(222)
        plt.axis('off')
        plt.title('Down')
        plt.imshow(data_down)
        plt.subplot(223)
        plt.axis('off')
        plt.title('Low')
        plt.imshow(data_low)
        plt.subplot(224)
        plt.axis('off')
        plt.title('High')
        plt.imshow(data_high)
        plt.show()
    return data_high, data_low


def read_raw_by_np(raw_path, shape=(120, 160), dtype='uint16'):
    raw_data = np.fromfile(raw_path, dtype=dtype)  # 按照uint16来读和处理
    # raw_data = raw_data // 4  # 右移2位相当于除以4后取整
    raw_data_real = raw_data
    # print('raw', np.mean(raw_data_real))
    imgData = raw_data_real.reshape(shape[0], shape[1])  # 利用numpy中array的reshape函数将读取到的数据进行重新排列。
    # imgData = imgData.astype('float32')
    # imgData = imgData / MAX  # imgData会自动变为float64类型
    # print(np.mean(imgData), imgData.dtype)
    return imgData


def cal_STDV(data):
    # data_high, data_low = get_high_low_by_fft(data, vis=True)
    data_high, data_low = get_high_low_by_pyramid(data, vis=True)
    # (mean_low, stddv_low) = cv2.meanStdDev(data_low)

    cv2.normalize(data_high, data_high, -19, 31, cv2.NORM_MINMAX)
    # cv::normalize(a, a, 0, 255, cv::NORM_MINMAX

    (mean_high, stddv_high) = cv2.meanStdDev(data_high)
    # print(mean_high)
    # raw_clean.tofile('High_high.raw')
    return stddv_high


if __name__ == '__main__':
    raw_dir = '/home/ipsg/code/sx/datasets/infread/raws/raw4/high'
    raw_paths = [i for i in Path(raw_dir).glob('*.raw')]
    raw_first = read_raw_by_np(str(raw_paths[0]))
    raw_shape = raw_first.shape
    raw_type = raw_first.dtype

    raw_clean = np.zeros(raw_shape, dtype='float64')
    for i in tqdm(range(len(raw_paths))):
        raw_tmp = read_raw_by_np(str(raw_paths[i]))
        raw_clean = raw_clean + raw_tmp
    raw_clean = raw_clean / len(raw_paths)
    raw_clean = raw_clean.astype(raw_type)
    # raw_clean.tofile('High_Clean.raw')
    # print('raw_clean_mean:',np.mean(raw_clean))

    sttv_clean = cal_STDV(raw_clean)
    sttv_noised = cal_STDV(raw_first)
    print('sttv_clean:{}\nsttv_noised:{}'.format(sttv_clean, sttv_noised))
