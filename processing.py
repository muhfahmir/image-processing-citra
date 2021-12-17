from math import cos, log, sin, sqrt, exp, pi

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import imageio
import random
# from imgaug import augmenters as iaa


def threshold(image, color_depth, threshold):
    """
        Thresholds an image to a given value.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new("RGB", image.size)

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            r, g, b = image.getpixel((x, y))
            if r and g and b < threshold:
                new_image.putpixel((x, y), (0, 0, 0))
            else:
                new_image.putpixel((x, y), (255, 255, 255))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def negative(image, color_depth):
    """
        Inverts an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new("RGB", image.size)

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            r, g, b = image.getpixel((x, y))
            new_image.putpixel((x, y), (255 - r, 255 - g, 255 - b))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def brightness(image, color_depth, brightness):
    """
        Adjusts the brightness of an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new("RGB", image.size)

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            r, g, b = image.getpixel((x, y))
            new_image.putpixel(
                (x, y), (r + brightness, g + brightness, b + brightness))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def rotate(image, color_depth, degrees):
    """
        Rotates an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")
    pixels0 = image.load()
    new_image = Image.new("RGB", image.size)

    x0 = image.size[0] // 2
    y0 = image.size[1] // 2

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            radians = (degrees * (22 / 7 / 180))

            # rotation formula with center of free rotation
            # source : https://homepages.inf.ed.ac.uk/rbf/HIPR2/rotate.htm
            x2 = (x - x0) * cos(radians) - (y - y0) * sin(radians) + x0
            y2 = (x - x0) * sin(radians) + (y - y0) * cos(radians) + y0

            if x2 >= image.size[0] or y2 >= image.size[1] or x2 < 0 or y2 < 0:
                new_image.putpixel((x, y), (0, 0, 0))
            else:
                new_image.putpixel((x, y), pixels0[int(x2), int(y2)])

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def flipping(image, color_depth, axis):
    """
        Flips an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new("RGB", image.size)

    if axis == 'Horizontal':
        for x in range(new_image.size[0]):
            for y in range(new_image.size[1]):
                new_image.putpixel((x, y), image.getpixel(
                    (new_image.size[0] - x - 1, y)))
    elif axis == 'Vertical':
        for x in range(new_image.size[0]):
            for y in range(new_image.size[1]):
                new_image.putpixel((x, y), image.getpixel(
                    (x, new_image.size[1] - y - 1)))
    else:
        for x in range(new_image.size[0]):
            for y in range(new_image.size[1]):
                new_image.putpixel((x, y), image.getpixel(
                    (new_image.size[0] - x - 1, new_image.size[1] - y - 1)))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def zooming(image, color_depth, scale):
    """
        Zooms an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new(
        "RGB", (image.size[0] * scale, image.size[1] * scale))

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            r, g, b = image.getpixel((x / scale, y / scale))
            new_image.putpixel((x, y), (r, g, b))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def shrinking(image, color_depth, scale):
    """
        Shrinks an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new(
        "RGB", (int(image.size[0] / scale), int(image.size[1] / scale)))

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            r, g, b = image.getpixel((x * scale, y * scale))
            new_image.putpixel((x, y), (r, g, b))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def blending(image1, image2, color_depth, color_depht2, alpha1, alpha2):
    """
        Blends two images.
    """
    if color_depth != 24:
        image1 = image1.convert("RGB")
    elif color_depht2 != 24:
        image2 = image2.convert("RGB")

    new_image = Image.new("RGB", image1.size)

    for x in range(new_image.size[0]):
        for y in range(new_image.size[1]):
            color1 = image1.getpixel((x, y))
            color2 = image2.getpixel((x, y))
            r = int((color1[0] * alpha1) + (color2[0] * alpha2))
            b = int((color1[1] * alpha1) + (color2[1] * alpha2))
            g = int((color1[2] * alpha1) + (color2[2] * alpha2))
            new_image.putpixel((x, y), (r, g, b))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def logarithmic(input_image, color_depth):
    if color_depth != 25:
        input_image = input_image.convert('RGB')
    C = 40
    output_image = Image.new(
        'RGB', (input_image.size[0], input_image.size[1]))
    pixels = output_image.load()
    for i in range(output_image.size[0]):
        for j in range(output_image.size[1]):
            r, g, b = input_image.getpixel((i, j))
            pixels[i, j] = (int(C*log(1+r)),
                            int(C*log(1+g)),
                            int(C*log(1+b)))
    if color_depth == 1:
        output_image = output_image.convert("1")
    elif color_depth == 8:
        output_image = output_image.convert("L")
    else:
        output_image = output_image.convert("RGB")

    return output_image


def translation(image, color_depth, shift):
    """
        Translates an image.
    """
    if color_depth != 24:
        image = image.convert("RGB")

    new_image = Image.new("RGB", image.size)

    start_m = shift[0]
    start_n = shift[1]

    if shift[0] < 0:
        start_m = 0
    if shift[1] < 0:
        start_n = 0

    for x in range(start_m, image.size[0]):
        for y in range(start_n, image.size[1]):
            new_x = x - shift[0]
            new_y = y - shift[1]

            if new_x >= image.size[0] or new_y >= image.size[1] or new_x < 0 or new_y < 0:
                new_image.putpixel((x, y), (0, 0, 0))
            else:
                new_image.putpixel((x, y), image.getpixel((new_x, new_y)))

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def filtering(image, color_depth, kernel_size=3, mode="Median"):
    if color_depth != 24:
        image = image.convert("RGB")

    red, green, blue = [], [], []
    indexer = kernel_size // 2
    new_image = Image.new("RGB", image.size)
    for x in range(indexer, image.size[0] - indexer):
        for y in range(indexer, image.size[1] - indexer):
            for z in range(kernel_size):
                if x + z - indexer < 0 or x + z - indexer > image.size[0] - 1:
                    for c in range(kernel_size):
                        red.append(0)
                        green.append(0)
                        blue.append(0)
                else:
                    if y + z - indexer < 0 or y + z - indexer > image.size[1] - 1:
                        red.append(0)
                        green.append(0)
                        blue.append(0)
                    else:
                        for k in range(kernel_size):
                            if y + k < image.size[1] and x + z < image.size[0]:
                                r, g, b = image.getpixel((x + z, y + k))
                                red.append(r)
                                green.append(g)
                                blue.append(b)
            if mode == "Median":
                red.sort()
                green.sort()
                blue.sort()
                r, g, b = (red[len(red) // 2],
                           green[len(green) // 2], blue[len(blue) // 2])
                new_image.putpixel((x, y), (r, g, b))
            elif mode == "Mean":
                r = sum(red) // len(red)
                g = sum(green) // len(green)
                b = sum(blue) // len(blue)
                new_image.putpixel((x, y), (r, g, b))
            red.clear()
            green.clear()
            blue.clear()

    if color_depth == 1:
        new_image = new_image.convert("1")
    elif color_depth == 8:
        new_image = new_image.convert("L")
    else:
        new_image = new_image.convert("RGB")

    return new_image


def edge_detection(image, color_depth, operator='Sobel'):
    """
        Applies an edge detection filter to an image.
        Edge Detection Operators are of two types:
            Gradient – based operator which computes first-order derivations
                       in a digital image like, Sobel, Prewitt, & Robert operator
            Gaussian – based operator which computes second-order derivations
                       in a digital image like, Canny edge detector, Laplacian of Gaussian
    """
    global kernel_x, kernel_y

    if color_depth != 24:
        image = image.convert("RGB")
    pixels1 = image.load()
    new_image = Image.new("RGB", image.size)

    # create pixel intensity as the average of RGB
    intensity = [[sum(pixels1[x, y]) / 3
                  for y in range(image.height)] for x in range(image.width)]

    if operator == "Sobel":
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    elif operator == "Prewitt":
        kernel_x = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        kernel_y = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    elif operator == "Robert":
        # kernel_x = [[1, 0], [0, -1]]
        # kernel_y = [[0, -1], [-1, 0]]
        kernel_x = [[0, 0, 0], [0, 1, 0], [0, 0, -1]]
        kernel_y = [[0, 0, 0], [0, 0, 1], [0, -1, 0]]

    elif operator == "Laplacian":
        return Image.fromarray(cv2.Laplacian(np.asarray(image.convert("L")), cv2.CV_64F)).convert("RGB")

    elif operator == "Canny":
        # kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        # kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        return Image.fromarray(auto_canny(image))

    # kernel canny

    for x in range(1, image.size[0] - 1):
        for y in range(1, image.size[1] - 1):
            magx, magy = 0, 0
            for i in range(3):
                for j in range(3):
                    xn = x + i - 1
                    yn = y + j - 1
                    magx += intensity[xn][yn] * kernel_x[i][j]
                    magy += intensity[xn][yn] * kernel_y[i][j]
            # calculate the magnitude of the gradient
            color = int(sqrt(magx ** 2 + magy ** 2))
            new_image.putpixel((x, y), (color, color, color))

    # if operator == "Canny":

    return new_image


# ----------------------------------------------------------------------------------------------------------------------

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(np.asarray(image), lower, upper)
    return edged


# ----------------------------------------------------------------------------------------------------------------------

# def noisy(image, noise_type, probability=0.2, scale=0.1):
#     if noise_type == "gaussian":
#         noise = iaa.AdditiveGaussianNoise(loc=0, scale=scale * 255)
#         new_image = noise.augment_image(image)
#         return new_image

#     elif noise_type == "salt&pepper":
#         noise = iaa.SaltAndPepper(p=probability)
#         new_image = noise.augment_image(image)
#         return new_image


# ----------------------------------------------------------------------------------------------------------------------

def applyGaussian(image, kernel):
    kernel_h = kernel.shape[0]
    kernel_w = kernel.shape[1]
    h = kernel_h // 2
    w = kernel_w // 2

    image = np.pad(image, pad_width=((h, h), (w, w)),
                   mode='edge').astype(np.float32)
    filtered_img = np.zeros(image.shape)

    for i in range(h, image.shape[0] - h):
        for j in range(w, image.shape[1] - w):
            x = image[i - h: i - h + kernel_h, j - w: j - w + kernel_w]
            x = x.flatten() * kernel.flatten()
            filtered_img[i][j] = x.sum()
    return filtered_img[h:-h, w:-w]


def getGaussianFilter(sigma, kernel_size):
    indexer = kernel_size // 2
    gaussian_filter = []
    for x in range(-indexer, indexer + 1):
        list = []
        for y in range(-indexer, indexer + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            list.append((1 / x1) * x2)
        gaussian_filter.append(list)
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    return gaussian_filter


def gaussian(x, y, sigma):
    denominator = sqrt(2 * pi * (sigma ** 2))
    exponent = -1 * ((x ** 2) + (y ** 2)) / (2 * (sigma ** 2))
    return (1 / denominator) * exp(exponent)


def gaussianBlur(image, color_depth, sigma, k_size):
    if color_depth != 24:
        image = image.convert("RGB")
    image = np.array(image)
    g_filter = getGaussianFilter(sigma, k_size)
    blur_image = np.zeros_like(image, np.float32)

    for c in range(3):
        blur_image[:, :, c] = applyGaussian(image[:, :, c], g_filter)
    return Image.fromarray(blur_image.astype(np.uint8))


# ----------------------------------------------------------------------------------------------------------------------

def dilate(img, kernel=np.ones((5, 5), np.uint8)):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    dilated_img = np.zeros(
        (img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if img[i + kernel_center[0], j + kernel_center[1]] == 255:
                dilated_img[i:i_, j:j_] = 255
    return dilated_img[:img_shape[0], :img_shape[1]]


def erode(img, kernel=np.ones((5, 5), np.uint8)):
    kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
    kernel_ones_count = kernel.sum()
    eroded_img = np.zeros(
        (img.shape[0] + kernel.shape[0] - 1, img.shape[1] + kernel.shape[1] - 1))
    img_shape = img.shape

    x_append = np.zeros((img.shape[0], kernel.shape[1] - 1))
    img = np.append(img, x_append, axis=1)

    y_append = np.zeros((kernel.shape[0] - 1, img.shape[1]))
    img = np.append(img, y_append, axis=0)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            i_ = i + kernel.shape[0]
            j_ = j + kernel.shape[1]
            if kernel_ones_count == (kernel * img[i:i_, j:j_]).sum() / 255:
                eroded_img[i + kernel_center[0], j + kernel_center[1]] = 255

    return eroded_img[:img_shape[0], :img_shape[1]]


# ----------------------------------------------------------------------------------------------------------------------
def rgb2gray(image):
    """
        Converts an RGB image to grayscale.
    """

    new_image = Image.new(size=image.size, mode="L")
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r, g, b = image.getpixel((x, y))
            new_image.putpixel((x, y), int(r * 0.299 + g * 0.587 + b * 0.114))

    return new_image


def rgb2hsv(image):
    """
        Converts an RGB image to HSV.
    """
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r, g, b = image.getpixel((x, y))
    return image.convert('HSV')


def contrast(input_image, value):
    imgNormal = imageio.imread(input_image)
    imgContrass = np.zeros(
        (imgNormal.shape[0], imgNormal.shape[1], 3), dtype=np.uint8)
    c = value  # 10 ~ 200
    c = float(c/100)
    for y in range(0, imgNormal.shape[0]):
        for x in range(0, imgNormal.shape[1]):
            r = int(float(imgNormal[y][x][0]) * c)
            g = int(float(imgNormal[y][x][1]) * c)
            b = int(float(imgNormal[y][x][2]) * c)
            if r < 0:
                r = 0
            if r > 255:
                r = 255
            if g < 0:
                g = 0
            if g > 255:
                g = 255
            if b < 0:
                b = 0
            if b > 255:
                b = 255
            imgContrass[y][x] = (r, g, b)
    return imgContrass
    # filename_out = f"images/result/contrast_1.bmp"
    # plt.imsave(filename_out, imgContrass)
    # window["-PREVIEW IMAGE OUTPUT-"].update(
    #     data=convert2bytes(filename_out, resize=img_box_size))


def sharpness(input_image, type_sharpness):
    imgNormal = imageio.imread(input_image)
    imgGrayscale = np.zeros(
        (imgNormal.shape[0], imgNormal.shape[1], 3), dtype=np.uint8)

    for y in range(0, imgNormal.shape[0]):
        for x in range(0, imgNormal.shape[1]):
            r = imgNormal[y][x][0]
            g = imgNormal[y][x][1]
            b = imgNormal[y][x][2]
            gr = (int(r) + int(g) + int(b)) / 3
            imgGrayscale[y][x] = (gr, gr, gr)

    imgSharpness = np.zeros(
        (imgNormal.shape[0], imgNormal.shape[1], 3), dtype=np.uint8)

    if type_sharpness == "Default":
        for y in range(1, imgNormal.shape[0] - 1):
            for x in range(1, imgNormal.shape[1] - 1):
                x1 = int(imgGrayscale[y - 1][x - 1][0])
                x2 = int(imgGrayscale[y][x - 1][0])
                x3 = int(imgGrayscale[y + 1][x - 1][0])
                x4 = int(imgGrayscale[y - 1][x][0])
                x5 = int(imgGrayscale[y][x][0])
                x6 = int(imgGrayscale[y + 1][x][0])
                x7 = int(imgGrayscale[y - 1][x + 1][0])
                x8 = int(imgGrayscale[y][x + 1][0])
                x9 = int(imgGrayscale[y + 1][x + 1][0])
                xt1 = int((x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9)
                xt2 = int(-x1 - (2 * x2) - x3 + x7 + (2 * x8) + x9)
                xt3 = int(-x1 - (2 * x4) - x7 + x3 + (2 * x6) + x9)
                xb = int(xt1 + xt2 + xt3)
                if xb < 0:
                    xb = -xb
                if xb > 255:
                    xb = 255
                imgSharpness[y][x] = (xb, xb, xb)
    elif type_sharpness == "2:1":
        for y in range(1, imgNormal.shape[0] - 1):
            for x in range(1, imgNormal.shape[1] - 1):
                x1 = int(imgGrayscale[y - 1][x - 1][0])
                x2 = int(imgGrayscale[y][x - 1][0])
                x3 = int(imgGrayscale[y + 1][x - 1][0])
                x4 = int(imgGrayscale[y - 1][x][0])
                x5 = int(imgGrayscale[y][x][0])
                x6 = int(imgGrayscale[y + 1][x][0])
                x7 = int(imgGrayscale[y - 1][x + 1][0])
                x8 = int(imgGrayscale[y][x + 1][0])
                x9 = int(imgGrayscale[y + 1][x + 1][0])
                xt1 = int((x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9)
                xt2 = int(-x1 - (2 * x2) - x3 + x7 + (2 * x8) + x9)
                xt3 = int(-x1 - (2 * x4) - x7 + x3 + (2 * x6) + x9)
                xb = int((2 * xt1) + xt2 + xt3)
                if xb < 0:
                    xb = -xb
                if xb > 255:
                    xb = 255
                imgSharpness[y][x] = (xb, xb, xb)
    elif type_sharpness == "1:2":
        for y in range(1, imgNormal.shape[0] - 1):
            for x in range(1, imgNormal.shape[1] - 1):
                x1 = int(imgGrayscale[y - 1][x - 1][0])
                x2 = int(imgGrayscale[y][x - 1][0])
                x3 = int(imgGrayscale[y + 1][x - 1][0])
                x4 = int(imgGrayscale[y - 1][x][0])
                x5 = int(imgGrayscale[y][x][0])
                x6 = int(imgGrayscale[y + 1][x][0])
                x7 = int(imgGrayscale[y - 1][x + 1][0])
                x8 = int(imgGrayscale[y][x + 1][0])
                x9 = int(imgGrayscale[y + 1][x + 1][0])
                xt1 = int((x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9)
                xt2 = int(-x1 - (2 * x2) - x3 + x7 + (2 * x8) + x9)
                xt3 = int(-x1 - (2 * x4) - x7 + x3 + (2 * x6) + x9)
                xb = int(xt1 + (2 * xt2) + (2 * xt3))
                if xb < 0:
                    xb = -xb
                if xb > 255:
                    xb = 255
                imgSharpness[y][x] = (xb, xb, xb)

    return imgSharpness


def noise(input_image, noise_type):
    imgNormal = imageio.imread(input_image)

    imgGrayscale = np.zeros(
        (imgNormal.shape[0], imgNormal.shape[1], 3), dtype=np.uint8)
    for y in range(0, imgNormal.shape[0]):
        for x in range(0, imgNormal.shape[1]):
            r = imgNormal[y][x][0]
            g = imgNormal[y][x][1]
            b = imgNormal[y][x][2]
            gr = (int(r) + int(g) + int(b)) / 3
            imgGrayscale[y][x] = (gr, gr, gr)

    imgNoise = np.zeros(
        (imgNormal.shape[0], imgNormal.shape[1], 3), dtype=np.uint8)
    if noise_type == "Gaussian":
        # Add noise
        for y in range(0, imgNormal.shape[0]):
            for x in range(0, imgNormal.shape[1]):
                xg = imgGrayscale[y][x][0]
                xb = xg
                nr = random.randint(0, 100)
                if nr < 20:
                    ns = random.randint(0, 256) - 128
                    xb = int(xg + ns)
                    if xb < 0:
                        xb = -xb
                    if xb > 255:
                        xb = 255
                imgNoise[y][x] = (xb, xb, xb)
    elif noise_type == "Speckle":
        # Add noise
        for y in range(0, imgNormal.shape[0]):
            for x in range(0, imgNormal.shape[1]):
                xg = imgGrayscale[y][x][0]
                xb = xg
                nr = random.randint(0, 100)
                if nr < 20:
                    xb = 0
                imgNoise[y][x] = (xb, xb, xb)
    elif noise_type == "Salt and Pepper":
        # Add noise
        for y in range(0, imgNormal.shape[0]):
            for x in range(0, imgNormal.shape[1]):
                xg = imgGrayscale[y][x][0]
                xb = xg
                nr = random.randint(0, 100)
                if nr < 20:
                    xb = 255
                imgNoise[y][x] = (xb, xb, xb)

    return imgNoise
