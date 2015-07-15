## Digit Detection & Recognition

### What is it?

Digit detection and recognition with AdaBoost and SVM.

![](preview.jpg)

### How it works

1. Train a cascade classifier for detection. The cascade classifier in `classifier/cascade.xml` is trained with 7000 positive samples and 9000 negative samples in 10 stages.
2. Train a SVM with the MNIST database.
3. Detect the digits in the image.
4. For each detected region, scale them to the same size as the samples in MNIST, then use the trained SVM to recognize(classify) the digits. For better results we can deskew the images with their momentum first, then use the HOG descriptors for testing.

### Dependencies

These scripts need python 2.7+ and the following libraries to work:

1. pillow(~2.8.1)
2. numpy(~1.9.0)
3. python-opencv(~2.4.11)
4. scikit-learn (~0.15.2)
The simplest way to install all of them is to install [python(x,y)](https://code.google.com/p/pythonxy/wiki/Downloads?tm=2).

If you can't install python(x,y), You can install python, numpy and python-opencv seperately, then install pip and pillow.

1. Install python. Just use the installer from [python's website](https://www.python.org/downloads/)
2. Install numpy. Just use the installer from [scipy's website](http://www.scipy.org/scipylib/download.html). (You don't need scipy to run this project, so you can just install numpy alone).
3. Install python-opencv. Download the release from [its sourceforge site](http://sourceforge.net/projects/opencvlibrary/files/). (Choose the release based on your operating system, then choose version 2.4.11). The executable is just an archive. Extract the files, then copy `cv2.pyd` to the `lib/site-packages` folder on your python installation path.
4. Install pip. Download [the script for installing pip](https://bootstrap.pypa.io/get-pip.py), open cmd (or termianl if you are using Linux/Mac OS X), go to the path where the downloaded script resides, and run `python get-pip.py`
5. Install pillow. Run `pip install pillow`. 
6. Install scikit-learn. Run `pip install scikit-learn`

If you are running the code under Linux/Mac OS X and the scripts throw `AttributeError: __float__`, make sure your pillow has jpeg support (consult [Pillow's document](http://pillow.readthedocs.org/en/latest/installation.html)) e.g. try:

```
sudo apt-get install libjpeg-dev
sudo pip uninstall pillow
sudo pip install pillow
```

If you have any problem installing the dependencies, contact the author.

### How to generate the results

Enter the `src` directory, run

```
python main.py
```

It will use images(`.jpg` only) under `test` directory to produce the results. The results will show up in `results` directory. Results generated with OpenCV will have `-cv` in its filename and results generated with sklearn will have `-sk` in its filename.


### Directory structure

```
.
├─ README.md
├─ doc (documentations, reports)
│   └── ...
├─ classifier (OpenCV cascade classifier)
│   ├── cascade.xml (the classifier parameter file)
│   └── ...
├─ MNIST (The MNIST database)
│   ├── train-images.idx3-ubyte
│   └── train-labels.idx1-ubyte
├─ test (test images)
│   └── ...
├─ results (the results)
│   └── ...
└─ src (the python source code)
    ├── detect.py (detection code)
    ├── load_labels.py (script to load MNIST data)
    ├── recognize.py (recognition code)
    └── main.py (generate the results)
```

### About

* [Github repository](https://github.com/joyeecheung/digit-detection-recognition)
* Author: Qiuyi Zhang
* Time: Jul. 2015