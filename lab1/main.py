import sys
import time
from typing import Tuple

import cv2.cv2 as cv2
import numpy as np

from itertools import product as itproduct


def main(filename: str = None, roi: Tuple[int, int, int, int] = None, kernel_size: int = 5):
    if not filename:
        if len(sys.argv) >= 1:
            filename = sys.argv[1]
        else:
            print("No filename specified")

    img: np.ndarray = cv2.imread(filename)
    if roi:
        img = img[roi[0]:roi[0] + roi[2], roi[1]:roi[1] + roi[3]]

    start = time.time()
    avg_filter_image = fast_avg_filt(img, kernel_size)
    print('fast', time.time() - start)
    # start = time.time()
    # avg_filter_image = avg_filt(img, kernel_size)
    # print('non_fast', time.time() - start)
    show_and_write_img(avg_filter_image, 'avg', 'avg.jpg')
    fast_med_image = fast_med_filt(img, kernel_size)
    show_and_write_img(fast_med_image, 'fast-med', 'fast-med.jpg')
    # med_filter_image = med_filt(img, kernel_size)
    # show_and_write_img(med_filter_image, 'med', 'med.jpg')
    hist_eq_image = hist_eq(img)
    show_and_write_img(hist_eq_image, 'hist-eq', 'hist-eq.jpg')

    img = add_border(img, 10)
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.imwrite('test_out.jpg', img)
    print("write successful")


def show_and_write_img(img: np.ndarray, title: str = 'img', filename: str = None):
    cv2.imshow(title, img)
    # cv2.waitKey(0)
    if filename:
        cv2.imwrite(filename, img)
        print("write to " + filename + " successfully")


def add_border(img: np.ndarray, padding: int = 1) -> np.ndarray:
    # create image with border
    img_with_border = np.zeros((img.shape[0] + padding * 2, img.shape[1] + padding * 2, img.shape[2]), img.dtype)
    # copy image
    img_with_border[padding:-padding, padding:-padding] = img
    for i in range(padding):
        # expand border
        img_with_border[i, padding:-padding] = img[0, :]
        img_with_border[padding:-padding, i] = img[:, 0]
        img_with_border[-1 - i, padding:-padding] = img[-1, :]
        img_with_border[padding:-padding, -1 - i] = img[:, -1]
    for i, j in itproduct(range(padding), range(padding)):
        # fill corner
        img_with_border[i, j] = img[0, 0]
        img_with_border[i, -1 - j] = img[0, -1]
        img_with_border[-1 - i, j] = img[-1, 0]
        img_with_border[-1 - i, -1 - j] = img[-1, -1]

    return img_with_border


def new_image(template: np.ndarray) -> np.ndarray:
    return np.zeros((template.shape[0], template.shape[1], template.shape[2]), template.dtype)


def avg_filt(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    padding = kernel_size // 2
    bordered_img = add_border(img, padding)
    result = new_image(img)
    for i, j in itproduct(range(result.shape[0]), range(result.shape[1])):
        result[i, j] = np.mean(bordered_img[i:i + kernel_size, j:j + kernel_size], axis=(0, 1))
    return result


def fast_avg_filt(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    padding = kernel_size // 2
    bordered_img = add_border(img, padding)
    prefix_mat = np.zeros((bordered_img.shape[0] + 1, bordered_img.shape[1] + 1, bordered_img.shape[2]), np.uint32)
    for i in range(bordered_img.shape[0]):
        for j in range(bordered_img.shape[1]):
            prefix_mat[i + 1, j + 1] = bordered_img[i, j] + prefix_mat[i, j + 1] + prefix_mat[i + 1, j] - prefix_mat[
                i, j]

    result = new_image(img)
    for i, j in itproduct(range(result.shape[0]), range(result.shape[1])):
        result[i, j] = (prefix_mat[i + kernel_size, j + kernel_size] - prefix_mat[i, j + kernel_size] - prefix_mat[
            i + kernel_size, j] + prefix_mat[i, j]) / kernel_size ** 2
    return result


def med_filt(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    padding = kernel_size // 2
    bordered_img = add_border(img, padding)
    result = new_image(img)
    for i, j in itproduct(range(result.shape[0]), range(result.shape[1])):
        # grey = np.sum(bordered_img[i:i + kernel_size, j:j + kernel_size], axis=2)
        # median = np.median(grey)
        # occurrences = np.where(grey == median)
        # index = occurrences[0][0], occurrences[1][0]
        # result[i, j] = bordered_img[i:i + kernel_size, j:j + kernel_size] [index]
        result[i, j] = np.median(bordered_img[i:i + kernel_size, j:j + kernel_size], axis=(0, 1))
    return result


def fast_med_filt(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    padding = kernel_size // 2
    bordered_img = add_border(img, padding)
    result = new_image(img)
    half_pixels = kernel_size ** 2 / 2

    hist: np.ndarray = np.zeros((img.shape[0], 256), np.uint16)
    mdn = np.zeros((img.shape[0]), np.uint16)
    ltmdn = np.zeros((img.shape[0]), np.uint16)

    # 创建左上角直方图
    for i, j in itproduct(range(kernel_size), range(kernel_size)):
        hist[0][bordered_img[i, j, 0]] += 1
    sum = hist[0][0]
    while sum < half_pixels:
        mdn[0] += 1
        sum += hist[0][mdn[0]]
    result[0, 0] = np.array([mdn[0]] * 3)
    ltmdn[0] = np.sum(hist[0, :mdn[0] + 1])

    # 创建第一列的初始直方图
    for row in range(1, img.shape[0]):
        hist[row] = hist[row - 1].copy()
        mdn[row] = mdn[row - 1]
        ltmdn[row] = ltmdn[row - 1]
        for i in range(kernel_size):
            hist[row][bordered_img[row - 1, i, 0]] -= 1
            hist[row][bordered_img[row + kernel_size - 1, i, 0]] += 1
            if bordered_img[row - 1, i, 0] <= mdn[row]:
                ltmdn[row] -= 1
            if bordered_img[row + kernel_size - 1, i, 0] <= mdn[row]:
                ltmdn[row] += 1

        while mdn[row] != 0 and ltmdn[row] - hist[row][mdn[row]] >= half_pixels:
            ltmdn[row] -= hist[row][mdn[row]]
            mdn[row] -= 1
        while ltmdn[row] < half_pixels:
            mdn[row] += 1
            ltmdn[row] += hist[row][mdn[row]]
        result[row, 0] = np.array([mdn[row]] * 3)

    # 对于每一行, 利用快速算法计算中值
    for row in range(img.shape[0]):
        for col in range(1, img.shape[1]):
            for i in range(kernel_size):
                hist[row][bordered_img[row + i, col - 1, 0]] -= 1
                hist[row][bordered_img[row + i, col + kernel_size - 1, 0]] += 1
                if bordered_img[row + i, col - 1, 0] <= mdn[row]:
                    ltmdn[row] -= 1
                if bordered_img[row + i, col + kernel_size - 1, 0] <= mdn[row]:
                    ltmdn[row] += 1

            while mdn[row] != 0 and ltmdn[row] - hist[row][mdn[row]] >= half_pixels:
                ltmdn[row] -= hist[row][mdn[row]]
                mdn[row] -= 1
            while ltmdn[row] < half_pixels:
                mdn[row] += 1
                ltmdn[row] += hist[row][mdn[row]]
            result[row, col] = np.array([mdn[row]] * 3)

    return result


def hist_eq(img: np.ndarray) -> np.ndarray:
    hist = np.zeros((256), np.uint32)
    for i, j in itproduct(range(img.shape[0]), range(img.shape[1])):
        hist[img[i, j, 0]] += 1
    cdf = np.cumsum(hist).astype(float)
    cdf_max = img.shape[0] * img.shape[1]
    cdf_min = cdf[np.nonzero(cdf)[0][0]]
    mapping = ((cdf - cdf_min) / (cdf_max - cdf_min) * 255).round().astype(np.uint8)
    result = new_image(img)
    result[:, :, 0] = mapping[img[:, :, 0]]
    result[:, :, 1] = result[:, :, 0]
    result[:, :, 2] = result[:, :, 0]
    return result


if __name__ == "__main__":
    # main('./landscape.jpg', (100, 100, 700, 1400))
    # main('./garage2.jpg', kernel_size=7)
    main('./salt_and_pepper.tif')
    # main('./ckt.tif')
    main('./aerial.tif')