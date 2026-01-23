
COLOR_DICT = {
        "background": "0, 0, 0",
        "image": "45, 255, 0",
        "text": "255, 243, 0",
        "margin": "0, 0, 255",
        "caption": "255, 100, 243",
        "table": "0, 255, 0",
        "pagenr": "0, 100, 15",
        "header": "255, 0, 0",
        "footer": "255, 255, 100",
        "line": "0, 100, 255",
        "separator": "23, 255, 45"
    }


MODERN_CLASSES = [
    "background",
    "image",
    "line",
    "header",
    "footer",
    "separator"
]


PERIG_CLASSES = [
    "background",
    "image",
    "line",
    "margin",
    "caption"
]

# some encoders, see full list here: https://smp.readthedocs.io/en/latest/encoders.html
BACKBONES = {
    "resnet" : "resnet34", # default, 21m parameters
    "efficientnet" : "timm-efficientnet-b5", # 28m parameters
    "restnest50" : "timm-restnest50d",
    "restnest100" : "timm-resnest101e", # 46m parameters
    "mit_b2" : "mit_b2", # 24m parameters
    "mit_b3" : "mit_b3",# 44m parameters
}