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