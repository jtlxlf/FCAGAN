# FCAGAN
This is the implementation for the paper:

**Cross-Lingual Font Style Transfer with Full-Domain Convolutional Attention**

By Hui-huang Zhao*, Tian-le Ji, Paul L. Rosin, Yu-Kun Lai, Wei-liang Meng, Yao-nan Wang

## Dependencies
Libarary
-------------
```
pytorch (>=1.7)
torchvision
dominate
visdom
tqdm
numpy
opencv-python  
scipy
sklearn
matplotlib  
pillow  
tensorboardX
scikit-image
scikit-learn
pytorch-fid
art-fid
```

Dataset
--------------
[方正字库] [汉仪字体] [蒙纳字体] [新蒂字体]  [造字工房], etc. provides free font download for non-commercial users.

# How to run

1. prepare dataset
   - Put your font files to a folder and character file to charset
        ```bash
        .
        ├── datasets
        │   └── test_unknown_content
        │       ├── chinese
        │       ├── english
        │       ├── source
        │   ├── test_unknown_style
        │       ├── chinese
        │       ├── english
        │       ├── source
        │   └── Train
        │       ├── chinese
        │       │   ├── chinesefont_0
        │       │   │   ├── char0.png
        │       │   │   ├── char1.png
        │       │   │   ├── ...
        │       │   ├── chinesefont_1
        │       │   ├── chinesefont_2
        │       │   ├── ...
        │       ├── english
        │       │   ├── englishfont_0
        │       │   │   ├── char0.png
        │       │   │   ├── char1.png
        │       │   │   ├── ...
        │       │   ├── englishfont_1
        │       │   ├── englishfont_2
        │       │   ├── ...
        │       ├── source
        │       │   ├── char0.png
        │       │   ├── char1.png
        │       │   ├── char2.png
        │       │   ├── ...
        ```

2. Train FCAGAN
   
3. Inference and evaluation
