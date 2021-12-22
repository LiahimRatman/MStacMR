# MStacMR

![alt text](https://github.com/LiahimRatman/MStacMR/blob/master/new_diag2-new_arcitecture.jpg?raw=true)

Model architecture is a modification of [StacMR project](https://github.com/AndresPMD/StacMR) with additions:
* custom image encoders
* custom OCR encoders
* some text experiments
* full inference pipeline

You can try application by [this url](http://34.88.238.34:8501/)

Project Organization
------------
    ├── config                          <- Config yaml files for train, evaluate, inference
    │
    ├── data
    │   ├── checkpoints_and_vocabs      <- Trained model checkpoints
    │   ├── precomputed_embeddings      <- Ocr and image embeddings
    │   ├── muse                        <- Muse pretrained model
    │   └── STACMR_train                <- Train dataset
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
    │                                      generated with `pip freeze > requirements.txt`
    │
    ├── src                             <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── README.md                       <- The top-level README for developers using this project.


--------
