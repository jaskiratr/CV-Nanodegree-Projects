# P2-Image-Captioning-Project

## 1. Setup COCO API
### MacOS/Linux
1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the COCO API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```
### Windows
To install COCO API follow steps listed here: https://github.com/philferriere/cocoapi, a fork maintained by [philferriere](https://github.com/philferriere/cocoapi).

## 2. Download Dataset
Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

```sh
📂 [Project-Folder]
└ 📂 opt
  └ 📂 cocoapi
    └ 📂 annotations
      └ 📄 captions_train2014.json
      └ 📄 captions_val2014.json
      └ 📄 image_info_test2014.json
      └ 📄 instances_train2014.json
      └ 📄 instances_val2014.json
      └ 📄 person_keypoints_train2014.json
      └ 📄 person_keypoints_val2014.json
    └ 📂 images
      └ 📂 test2014
        └ 📄 COCO_test2014_000000000001.jpg 
        └ 📄 ...
      └ 📂 train2014
        └ 📄 COCO_train2014_000000000009.jpg
        └ 📄 ...
```

The project is structured as a series of Jupyter notebooks that are designed to be completed in sequential order (`0_Dataset.ipynb, 1_Preliminaries.ipynb, 2_Training.ipynb, 3_Inference.ipynb`).

## 3. Download Models (Optional)
Pre-trained model trained for 1 epoch can be downloaded from below.
- [encoder-1.pkl](https://www.dropbox.com/s/c7x99fgnbhowrcy/encoder-1.pkl?dl=0)
- [decoder-1.pkl](https://www.dropbox.com/s/spmzrta00wsfngx/decoder-1.pkl?dl=0)

Store the downloaded models in `models` folder.
```sh
📂 [Project-Folder]
└ 📂 models
    └ 📄 encoder-1.pkl
    └ 📄 decoder-1.pkl
```