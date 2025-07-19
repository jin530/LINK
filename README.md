## Requirements

you can use the following command to install the environment
```bash
conda create -n link python=3.8
conda install pytorch==1.11.0 -c pytorch
pip install -r requirements.txt
```

## Datasets
make `dataset` folder and unzip datasets
$DATASET$: (`diginetica`, `retailrocket`, `yoochoose`, `dressipi`, `tmall`, `lastfm`)
```bash
for DATASET in diginetica retailrocket yoochoose dressipi tmall lastfm
do
    tar -zxvf $DATASET.tar.gz
done
```

## Reproduction

1. run `run_core.sh` to get core_trm results
2. run `run_link.sh` to get link results

## Citation

Please cite our paper:
```
@inproceedings{10.1145/3726302.3730024,
author = {Choi, Minjin and Lee, Sunkyung and Park, Seongmin and Lee, Jongwuk},
title = {Linear Item-Item Models with Neural Knowledge for Session-based Recommendation},
year = {2025},
isbn = {9798400715921},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3726302.3730024},
doi = {10.1145/3726302.3730024},
booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1666–1675},
numpages = {10},
location = {Padua, Italy},
series = {SIGIR '25}
}
```
