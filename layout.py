import PySimpleGUI as sg
sg.theme("DarkTeal9")

left_col = [
    [sg.Text("Open Image Folder :")],
    [
        sg.In(size=(20, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse()
    ],
    [sg.Text("Choose an images from list : ")],
    [sg.Listbox(values=[], enable_events=True,
                size=(25, 20), key="-FILE LIST-")],

    [sg.Text("Image Information : ")],
    [sg.Text(size=(20, 1),
             key="-IMAGE SIZE-")],
    [sg.Text(size=(20, 1),
             key="-IMAGE COLOR DEPTH-")],
]

image_input = [
    [sg.Text("Image Input:")],
    [sg.Text(size=(40, 1), key="-IMAGE PATH-")],
    [sg.Image(key="-PREVIEW IMAGE INPUT-")]
]

image_input2 = [
    [sg.Text("Image Input 2:", key="-TEXT IMAGE 2-")],
    [sg.Text(size=(40, 1), key="-IMAGE PATH2-")],
    [sg.Image(key="-PREVIEW IMAGE INPUT2-")]
]

image_output = [
    [sg.Text("Image Output:")],
    [sg.Text(size=(40, 1), key="-TYPE PROCESSING-")],
    [sg.Image(key="-PREVIEW IMAGE OUTPUT-")]
]

value_process = [
    [sg.Button("Process", size=(15, 1), key="-BUTTON PROCESS-")],
    [sg.Button("Show Histogram", size=(
        15, 1), key="-BUTTON HISTOGRAM-")],
    [sg.Button("Reset", size=(15, 1), key="-BUTTON RESET-")]
]

layout = [
    [
        sg.Column(left_col),
        sg.VSeperator(color="#D1D5DB"),
        sg.Column(image_input),
        sg.Column(image_input2, key="-COLUMN IMAGE INPUT 2-", visible=False),
        sg.Column(value_process),
        # sg.Frame(layout=[
        #     [sg.Button("Process", size=(15, 1), key="-BUTTON PROCESS-")],
        #     [sg.Button("Show Histogram", size=(
        #         15, 1), key="-BUTTON HISTOGRAM-")],
        #     [sg.Button("Reset", size=(15, 1), key="-BUTTON RESET-")],
        #     # [sg.Button("Show Histogram 2", size=(15, 1))],
        # ], title="PROCESS", relief=sg.RELIEF_SUNKEN, size=(140, 140)),
        sg.Column(image_output, key="-COLUMN IMAGE OUTPUT-"),
    ],
    [
        [
            sg.Frame(layout=[
                [
                    sg.Button("Image Thresholding", size=(20, 1),
                              key="-BUTTON IMAGE THRESHOLDING-"),
                    sg.Button("Image Negative", size=(20, 1),
                              key="-BUTTON IMAGE NEGATIVE-"),
                    sg.Button("Image Brightness", size=(20, 1),
                              key="-BUTTON IMAGE BRIGHTNESS-"),
                    sg.Button("Image Rotation", size=(20, 1),
                              key="-BUTTON IMAGE ROTATION-"),
                    sg.Button("Image Mirroring", size=(20, 1),
                              key="-BUTTON IMAGE FLIPPING-"),
                    sg.Button("Image Zooming", size=(20, 1),
                              key="-BUTTON IMAGE ZOOMING-"),

                ],
                [
                    sg.Button("Image Shrinking", size=(20, 1),
                              key="-BUTTON IMAGE SHRINKING-"),
                    sg.Button("Image Blending", size=(20, 1),
                              key="-BUTTON IMAGE BLENDING-"),
                    sg.Button("Image Logarithmic", size=(20, 1),
                              key="-BUTTON IMAGE LOGARITHMIC-"),
                    sg.Button("Image Translation", size=(20, 1),
                              key="-BUTTON IMAGE TRANSLATION-"),
                    sg.Button("Image Smoothing", size=(20, 1),
                              key="-BUTTON STATISTICAL FILTERING-"),
                    sg.Button("Image Gaussian", size=(20, 1),
                              key="-BUTTON IMAGE GAUSSIAN-"),
                ],
                [
                    sg.Button("Image Noise", size=(20, 1),
                              key="-BUTTON IMAGE NOISE-"),
                    sg.Button("Image Grayscale", size=(20, 1),
                              key="-BUTTON IMAGE RGB2GRAY-"),
                    sg.Button("Image Edge Detection", size=(20, 1),
                              key="-BUTTON IMAGE EDGE DETECTION-"),
                    sg.Button("Image Morphology", size=(20, 1),
                              key="-BUTTON IMAGE MORPHOLOGY-"),
                    sg.Button("Image Contrast", size=(20, 1),
                              key="-BUTTON IMAGE CONTRAST-"),
                    sg.Button("Image Sharpness", size=(20, 1),
                              key="-BUTTON IMAGE SHARPNESS-"),
                ]
            ], title="FEATURE", relief=sg.RELIEF_SUNKEN),
            sg.Frame(layout=[
                [
                    sg.Column(layout=[[
                        sg.Text("Threshold Value : "),
                        sg.Slider(range=(0, 260), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE THRESHOLDING-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE THRESHOLDING-"),
                    sg.Column(layout=[[
                        sg.Text("Brightness Value : "),
                        sg.Slider(range=(-255, 255), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE BRIGHTNESS-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE BRIGHTNESS-"),

                    sg.Column(layout=[[
                        sg.Text("Degrees : "),
                        sg.Slider(range=(0, 360), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE ROTATION-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE ROTATION-"),

                    sg.Column(layout=[[
                        sg.Text("Zoom : "),
                        sg.Slider(range=(2, 4), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE ZOOMING-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE ZOOMING-"),

                    sg.Column(layout=[[
                        sg.Text("Shrink : "),
                        sg.Slider(range=(2, 4), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE SHRINKING-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE SHRINKING-"),

                    sg.Column(layout=[[
                        sg.Text("Image 2 : "),
                        sg.In(size=(20, 0), enable_events=True,
                              key="-INPUT IMAGE2-"),
                        sg.FileBrowse(enable_events=True,
                                      key="-FILE BROWSE IMAGE2-"),
                    ]], visible=False, key="-COLUMN IMAGE BROWSE 2-"),

                    sg.Column(layout=[[
                        sg.Text("Alpha Image 1: (0.0 - 1.0)",
                                key="-TEXT IMAGE ALPHA-"),
                        sg.In(size=(20, 1), enable_events=True,
                              key="-INPUT IMAGE ALPHA-"),
                    ]], visible=False, key="-COLUMN IMAGE ALPHA-"),

                    sg.Column(layout=[[
                        sg.Text("Alpha Image 2: (0.0 - 1.0)",
                                key="-TEXT IMAGE2 ALPHA-"),
                        sg.In(size=(20, 1), enable_events=True,
                              key="-INPUT IMAGE2 ALPHA-"),
                    ]], visible=False, key="-COLUMN IMAGE2 ALPHA-"),

                    sg.Column(layout=[[
                        sg.Text("Select Axis : "),
                        sg.Drop(
                            values=("Horizontal", "Vertical",
                                    "Vertical Horizontal"),
                            auto_size_text=True, default_value="Vertical",
                            key="-DROP DOWN IMAGE FLIPPING-"),
                    ]], visible=False, key="-COLUMN DROP DOWN IMAGE FLIPPING-"),

                    sg.Column(layout=[[
                        sg.Text("x : "),
                        sg.In(size=(20, 1), enable_events=True,
                              key="-INPUT HORIZONTAL TRANSLATION-"),
                    ]], visible=False, key="-COLUMN IMAGE TRANSLATION X-"),

                    sg.Column(layout=[[
                        sg.Text("y : "),
                        sg.In(size=(20, 1), enable_events=True,
                              key="-INPUT VERTICAL TRANSLATION-"),
                    ]], visible=False, key="-COLUMN IMAGE TRANSLATION Y-"),

                    sg.Column(layout=[
                        [
                            sg.Text("Kernel : "),
                            sg.Drop(values=("3x3", "5x5", "7x7", "9x9"),
                                    auto_size_text=True, default_value="3x3",
                                    key="-DROP DOWN GAUSSIAN KERNEL SIZE-"),
                        ],
                        [
                            sg.Text("Sigma Size : "),
                            sg.In(size=(20, 1), enable_events=True,
                                  key="-INPUT SIGMA GAUSSIAN BLUR-"),
                        ]
                    ], visible=False, key="-COLUMN IMAGE GAUSSIAN BLUR FILTERING-"),

                    sg.Column(layout=[[
                        sg.Text("Filter : "),
                        sg.Drop(values=("Sobel", "Prewitt",
                                        "Canny",
                                        "Robert",
                                        "Laplacian"
                                        ),
                                auto_size_text=True, default_value="Sobel",
                                key="-DROP DOWN EDGE DETECTION TYPE-"),
                    ]], visible=False, key="-COLUMN IMAGE EDGE DETECTION-"),

                    sg.Column(layout=[[
                        sg.Text("Gray : "),
                        sg.Drop(values=("Default", "2:1",
                                        "1:2",
                                        ),
                                auto_size_text=True, default_value="Default",
                                key="-DROP DOWN SHARPNESS TYPE-"),
                    ]], visible=False, key="-COLUMN IMAGE SHARPNESS-"),

                    sg.Column(layout=[[
                        sg.Text("Noise : "),
                        sg.Drop(values=("Gaussian", "Speckle",
                                        "Salt and Pepper",
                                        ),
                                auto_size_text=True, default_value="Gaussian",
                                key="-DROP DOWN NOISE TYPE-"),
                    ]], visible=False, key="-COLUMN IMAGE NOISE-"),

                    sg.Column(layout=[[
                        sg.Text("Type : "),
                        sg.Drop(values=("Erode",
                                        "Dilate",
                                        "Open",
                                        "Close"),
                                auto_size_text=True, default_value="Erode",
                                key="-DROP DOWN MORPHOLOGY TYPE-"),
                    ]], visible=False, key="-COLUMN IMAGE MORPHOLOGY-"),

                    sg.Column(layout=[[
                        sg.Text("Contrast Value : "),
                        sg.Slider(range=(0, 200), size=(19, 20),
                                  orientation="h",
                                  key="-SLIDER IMAGE CONTRAST-",
                                  default_value=0),
                    ]], visible=False, key="-COLUMN SLIDER IMAGE CONTRAST-"),

                    sg.Column(layout=[
                        [
                            sg.Text("Filter : "),
                            sg.Drop(values=("Median", "Mean"),
                                    auto_size_text=True, default_value="Median",
                                    key="-DROP DOWN STATISTICAL FILTERING-"),
                        ],
                        [
                            sg.Text("Kernel : "),
                            sg.Drop(values=("3x3", "5x5", "7x7", "9x9"),
                                    auto_size_text=True, default_value="3x3",
                                    key="-DROP DOWN MEDIAN KERNEL SIZE-"),
                        ]
                    ], visible=False, key="-COLUMN STATISTICAL FILTERING-"),

                ]
            ], title="VALUE INPUT", relief=sg.RELIEF_SUNKEN, size=(300, 130)),

        ],
    ],
]

# --------------------------------- Create Window ---------------------------------
window = sg.Window("Proyek Akhir Pengolahan Citra", layout,
                   resizable=False,  location=(10, 0))
