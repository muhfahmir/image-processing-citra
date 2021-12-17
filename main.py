import os.path

from PIL import ImageFilter

from layout import *
from utils import convert2bytes, getRed, getGreen, getBlue
from processing import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def show_histogram(image, title1, image2, title2):
    color = ('b', 'g', 'r')
    color2 = ('b', 'g', 'r')
    plt.figure(0)
    plt.cla()
    for channel, col in enumerate(color):
        histr = cv2.calcHist([image], [channel], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.title(f"{title1} Histogram")

    plt.figure(1)
    plt.cla()
    for channel, col in enumerate(color2):
        histr2 = cv2.calcHist([image2], [channel], None, [256], [0, 256])
        plt.plot(histr2, color=col)
        plt.xlim([0, 256])

    plt.title(f"{title2} Histogram")

    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='left', fill='both', expand=1)
        return figure_canvas_agg

    # ------------------------------- Beginning of GUI CODE -------------------------------

    # Define the window layout
    layout = [[sg.Canvas(key='-CANVAS-')]]

    # create the form and show it without the plot
    window = sg.Window('Histogram', layout, location=(0, 0), finalize=True, element_justification='center',
                       font='Helvetica 18', resizable=True)

    # Add the plot to the window
    draw_figure(window['-CANVAS-'].TKCanvas, plt.figure(0))
    draw_figure(window['-CANVAS-'].TKCanvas, plt.figure(1))


def main():
    global input_image, input_image2, mode_to_color_depth, color_depth, color_depth2, img_box_size, full_path, default_filename, type_processing, output_image, output_image_temp, filename_out

    while True:

        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == sg.WIN_CLOSED or event == "Exit":
            break

        if event == "-FOLDER-":
            folder = values["-FOLDER-"]
            try:
                file_list = os.listdir(folder)
            except Exception as E:
                file_list = []
                print(f"** Error {E} **")

            extension = (".bmp", ".jpg", "jpeg", ".pgm",
                         ".ppm", ".pbm", ".png", ".raw")
            fnames = [f for f in file_list if os.path.isfile(
                os.path.join(folder, f)) and f.lower().endswith(extension)]
            window["-FILE LIST-"].update(fnames)

        elif event == "-FILE LIST-":
            try:
                img_box_size = int(450), int(450)
                full_path = os.path.join(
                    values["-FOLDER-"], values["-FILE LIST-"][0])
                filename = values["-FILE LIST-"][0].split(".")[0]
                filename_out = full_path

                default_filename = filename
                window["-IMAGE PATH-"].update(values["-FILE LIST-"][0])
                window["-PREVIEW IMAGE INPUT-"].update(
                    data=convert2bytes(full_path, resize=img_box_size))
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(full_path, resize=img_box_size))

                input_image = Image.open(full_path)
                image_width, image_height = input_image.size
                window["-IMAGE SIZE-"].update("Resolution : " +
                                              str(image_width) + " x " + str(image_height))
                mode_to_color_depth = {"1": 1, "L": 8, "P": 8, "RGB": 24, "RGBA": 32, "CMYK": 32, "YCbCr": 24,
                                       "LAB": 24,
                                       "HSV": 24, "I": 32, "F": 32}
                color_depth = mode_to_color_depth[input_image.mode]
                window["-IMAGE COLOR DEPTH-"].update(
                    "Color Depth: " + str(color_depth))
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE THRESHOLDING-":
            try:
                window["-TYPE PROCESSING-"].update("Image Thresholding")
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=True)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE NEGATIVE-":
            try:
                window["-TYPE PROCESSING-"].update("Image Negative")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                output_image = negative(input_image, color_depth)
                filename_out = "images/result/negative.bmp"
                output_image.save(filename_out)
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(filename_out, resize=img_box_size))
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE BRIGHTNESS-":
            try:
                window["-TYPE PROCESSING-"].update("Image Brightness")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(visible=True)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE ROTATION-":
            try:
                window["-TYPE PROCESSING-"].update("Image Rotating")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=True)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE FLIPPING-":
            try:
                window["-TYPE PROCESSING-"].update("Image Flipping")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=True)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE ZOOMING-":
            try:
                window["-TYPE PROCESSING-"].update("Image Zooming")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=True)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE SHRINKING-":
            try:
                window["-TYPE PROCESSING-"].update("Image Shrinking")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=True)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE BLENDING-":
            try:
                window["-TYPE PROCESSING-"].update("Image Blending")
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=True)
                window["-COLUMN IMAGE ALPHA-"].update(visible=True)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=True)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-INPUT IMAGE2-":
            try:
                window["-COLUMN IMAGE INPUT 2-"].update(visible=True)
                window["-IMAGE PATH2-"].update(value=values["-INPUT IMAGE2-"])
                input_image2 = Image.open(values["-INPUT IMAGE2-"])
                window["-PREVIEW IMAGE INPUT2-"].update(
                    data=convert2bytes(values["-INPUT IMAGE2-"], resize=img_box_size))
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE LOGARITHMIC-":
            try:
                window["-TYPE PROCESSING-"].update("Image Logarithmic")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

                output_image = logarithmic(input_image, color_depth)
                filename_out = "images/result/logarithmic.bmp"
                output_image.save(filename_out)
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(filename_out, resize=img_box_size))
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE TRANSLATION-":
            try:
                window["-TYPE PROCESSING-"].update("Image Translation")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=True)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=True)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE EDGE DETECTION-":
            try:
                window["-TYPE PROCESSING-"].update("Image Edge Detection")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=True)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE RGB2GRAY-":
            try:
                window["-TYPE PROCESSING-"].update("Image RGB2GRAY")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)

                output_image = rgb2gray(input_image)
                filename_out = f"images/result/rgb2grayscale.bmp"
                output_image.save(filename_out)
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(filename_out, resize=img_box_size))
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)
            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE RGB2HSV-":
            try:
                window["-TYPE PROCESSING-"].update("Image RGB2GRAY")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

                output_image = rgb2hsv(input_image)
                filename_out = f"images/result/rgb2hsv.jpg"
                output_image.save(filename_out)
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(filename_out, resize=img_box_size))

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE GAUSSIAN-":
            try:
                window["-TYPE PROCESSING-"].update("Image Gaussian Blur")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=True)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON STATISTICAL FILTERING-":
            try:
                window["-TYPE PROCESSING-"].update("Statistical Filtering")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=True)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE NOISE-":
            try:
                window["-TYPE PROCESSING-"].update("Image Noise")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=True)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE MORPHOLOGY-":
            try:
                window["-TYPE PROCESSING-"].update("Image Morphology")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=True)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=True)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON HISTOGRAM-":
            try:
                # input image
                img = cv2.imread(full_path, -1)
                plt.figure(1)
                color = ('b', 'g', 'r')
                for channel, col in enumerate(color):
                    histr = cv2.calcHist(
                        [img], [channel], None, [256], [0, 256])
                    plt.plot(histr, color=col)
                    plt.xlim([0, 256])
                plt.title('Input Image Histogram')

                plt.figure(2)
                # output image
                img2 = cv2.imread(filename_out, -1)
                color = ('b', 'g', 'r')
                for channel, col in enumerate(color):
                    histr2 = cv2.calcHist(
                        [img2], [channel], None, [256], [0, 256])
                    plt.plot(histr2, color=col)
                    plt.xlim([0, 256])
                plt.title('Output Image Histogram')
                plt.show()
            except Exception as E:
                print(f"** Error {E} **")
                pass
        elif event == "-BUTTON RESET-":
            try:
                output_image = input_image
                window["-PREVIEW IMAGE OUTPUT-"].update(
                    data=convert2bytes(full_path, resize=img_box_size))
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-INPUT IMAGE2-"].update(value="")
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)
                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass
        # added
        elif event == "-BUTTON IMAGE CONTRAST-":
            try:
                window["-TYPE PROCESSING-"].update("Image Contrast")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)

                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=True)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=False)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON IMAGE SHARPNESS-":
            try:
                window["-TYPE PROCESSING-"].update("Image Sharpness")
                window["-COLUMN IMAGE INPUT 2-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE THRESHOLDING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE BRIGHTNESS-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ROTATION-"].update(visible=False)
                window["-COLUMN DROP DOWN IMAGE FLIPPING-"].update(
                    visible=False)
                window["-COLUMN SLIDER IMAGE ZOOMING-"].update(visible=False)
                window["-COLUMN SLIDER IMAGE SHRINKING-"].update(visible=False)
                window["-COLUMN IMAGE BROWSE 2-"].update(visible=False)
                window["-COLUMN IMAGE ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE2 ALPHA-"].update(visible=False)
                window["-COLUMN IMAGE EDGE DETECTION-"].update(visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN STATISTICAL FILTERING-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION X-"].update(visible=False)
                window["-COLUMN IMAGE TRANSLATION Y-"].update(visible=False)
                window["-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"].update(
                    visible=False)
                window["-COLUMN IMAGE NOISE-"].update(visible=False)
                window["-COLUMN IMAGE MORPHOLOGY-"].update(visible=False)

                # tambahan
                window["-COLUMN SLIDER IMAGE CONTRAST-"].update(visible=False)
                window["-COLUMN IMAGE SHARPNESS-"].update(visible=True)

            except Exception as E:
                print(f"** Error {E} **")
                pass

        elif event == "-BUTTON PROCESS-":
            try:
                type_processing = window["-TYPE PROCESSING-"].get()

                if type_processing == "Image Thresholding":
                    thresholding_value = int(
                        values["-SLIDER IMAGE THRESHOLDING-"])
                    output_image = threshold(
                        input_image, color_depth, thresholding_value)
                    filename_out = f"images/result/thresholding_{thresholding_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Brightness":
                    brightness_value = int(values["-SLIDER IMAGE BRIGHTNESS-"])
                    output_image = brightness(
                        input_image, color_depth, brightness_value)
                    filename_out = f"images/result/brightness_{brightness_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Rotating":
                    rotate_value = int(values["-SLIDER IMAGE ROTATION-"])
                    output_image = rotate(
                        input_image, color_depth, rotate_value)
                    filename_out = f"images/result/rotating_{rotate_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Flipping":
                    flipping_value = values["-DROP DOWN IMAGE FLIPPING-"]
                    output_image = flipping(
                        input_image, color_depth, flipping_value)
                    filename_out = f"images/result/flipping_{flipping_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Zooming":
                    zoom_value = int(values["-SLIDER IMAGE ZOOMING-"])
                    output_image = zooming(
                        input_image, color_depth, zoom_value)
                    filename_out = f"images/result/zooming_{zoom_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out))

                elif type_processing == "Image Shrinking":
                    shrink_value = int(values["-SLIDER IMAGE SHRINKING-"])
                    output_image = shrinking(
                        input_image, color_depth, shrink_value)
                    filename_out = f"images/result/shrinking_{shrink_value}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out))

                elif type_processing == "Image Blending":
                    input_image_alpha = float(values["-INPUT IMAGE ALPHA-"])
                    input_image2_alpha = float(values["-INPUT IMAGE2 ALPHA-"])
                    color_depth2 = mode_to_color_depth[input_image2.mode]
                    output_image = blending(
                        input_image, input_image2, color_depth, color_depth2, input_image_alpha, input_image2_alpha)
                    filename_out = f"images/result/blending.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Translation":
                    translation_x = int(
                        values["-INPUT HORIZONTAL TRANSLATION-"])
                    translation_y = int(values["-INPUT VERTICAL TRANSLATION-"])
                    shift = (translation_x, translation_y)
                    output_image = translation(input_image, color_depth, shift)
                    filename_out = f"images/result/translation_{translation_x}_{translation_y}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Edge Detection":
                    filter_type = values["-DROP DOWN EDGE DETECTION TYPE-"]
                    output_image = edge_detection(
                        input_image, color_depth, filter_type)
                    filename_out = f"images/result/{default_filename}_ED_{filter_type}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Gaussian Blur":
                    sigma = float(values["-INPUT SIGMA GAUSSIAN BLUR-"])
                    kernel = int(values["-DROP DOWN GAUSSIAN KERNEL SIZE-"][0])
                    output_image = gaussianBlur(
                        input_image, color_depth, sigma, kernel)
                    filename_out = f"images/result/{default_filename}_GaussianBlur.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Statistical Filtering":
                    type = values["-DROP DOWN STATISTICAL FILTERING-"]
                    kernel_size = int(
                        values["-DROP DOWN MEDIAN KERNEL SIZE-"][0])
                    output_image = filtering(
                        input_image, color_depth, kernel_size, type)
                    filename_out = f"images/result/{default_filename}_{type}{kernel_size}x{kernel_size}.bmp"
                    output_image.save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Morphology":
                    type = values["-DROP DOWN MORPHOLOGY TYPE-"]

                    img = np.asarray(input_image.convert("L"))
                    (thresh, img) = cv2.threshold(img, 128,
                                                  255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    if type == "Dilate":
                        output_image_temp = dilate(img)
                    elif type == "Erode":
                        output_image_temp = erode(img)
                    elif type == "Open":
                        output_image_temp = dilate(erode(img))
                    elif type == "Close":
                        output_image_temp = erode(dilate(img))

                    output_image = Image.fromarray(output_image_temp)
                    filename_out = f"images/result/{default_filename}_{type}.bmp"
                    output_image.convert("RGB").save(filename_out)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                # added
                elif type_processing == "Image Contrast":
                    contrast_value = int(values["-SLIDER IMAGE CONTRAST-"])
                    output_image = contrast(
                        full_path, contrast_value)
                    filename_out = f"images/result/contrast{contrast_value}.bmp"
                    plt.imsave(filename_out, output_image)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Sharpness":
                    filter_type = values["-DROP DOWN SHARPNESS TYPE-"]
                    output_image = sharpness(
                        full_path, filter_type)
                    filename_out = f"images/result/sharpness_{filter_type}.bmp"
                    plt.imsave(filename_out, output_image)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

                elif type_processing == "Image Noise":
                    noise_type = values["-DROP DOWN NOISE TYPE-"]
                    # print(noise_type)
                    output_image = noise(
                        full_path, noise_type)
                    filename_out = f"images/result/noise_{noise_type}.bmp"
                    plt.imsave(filename_out, output_image)
                    window["-PREVIEW IMAGE OUTPUT-"].update(
                        data=convert2bytes(filename_out, resize=img_box_size))

            except Exception as E:
                print(f"** Error {E} **")
                pass

    # --------------------------------- Close & Exit ---------------------------------
    window.close()


if __name__ == '__main__':
    main()
