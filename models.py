upsamplers = {
    "realesr-general-x4v3": {
        "name": "General - v3",
        "weights": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "path": "realesr-general-x4v3.pth",
        "net": "SRVGGNetCompact",
        "initArgs": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_conv": 32,
            "upscale": 4,
            "act_type": "prelu",
        },
        "netscale": 4,
    },
}

face_enhancers = {
    "GFPGAN": {
        "name": "GFPGAN",
        "weights": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "path": "weights/GFPGANv1.4.pth",
    },
}

models_by_type = {
    "upsamplers": upsamplers,
    "face_enhancers": face_enhancers,
}

