# minMusicTransformer
Transfer Learning on a Transformer based on https://github.com/salu133445/mmt

## Content


- [Prerequisites](#prerequisites)
- [Preprocessing](#preprocessing)
  - [Preprocessed Datasets](#preprocessed-datasets)
  - [Preprocessing Scripts](#preprocessing-scripts)
- [Training](#training)
  - [Pretrained Models](#pretrained-models)
  - [Training Scripts](#training-scripts)
- [Evaluation](#evaluation)
- [Generation (Inference)](#generation-inference)
- [Citation](#citation)

## Prerequisites

We recommend using Conda. You can create the environment with the following command.

```sh
conda env create -f environment.yml
```

## Preprocessing

### Preprocessed Datasets

Some preprocessed datasets can be found [here](https://drive.google.com/drive/folders/1owWu-Ne8wDoBYCFiF9z11fruJo62m_uK?usp=share_link). You can use [gdown](https://github.com/wkentaro/gdown) to download them via command line as follows.

```sh
gdown --id 1owWu-Ne8wDoBYCFiF9z11fruJo62m_uK --folder
```

### Preprocessing Scripts

__You can skip this section if you download the preprocessed datasets.__

#### Step 1 -- Download the datasets

Please download the [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html). You may download it via command line as follows.

```sh
wget https://qsdfo.github.io/LOP/database/SOD.zip
```

We also support the following two datasets:

- [Lakh MIDI Dataset (LMD)](https://qsdfo.github.io/LOP/database.html):

  ```sh
  wget http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
  ```

- [SymphonyNet Dataset](https://symphonynet.github.io/):

  ```sh
  gdown https://drive.google.com/u/0/uc?id=1j9Pvtzaq8k_QIPs8e2ikvCR-BusPluTb&export=download
  ```

## Splitting and Preparing Data

We are assuming data is already split into a testing and training set
Use the convert_extract_load function, to convert data from midi to json to npy and load them into a dataset.

## Training

More on that soon

### Pretrained Models

More pretrained models can be found [here](https://drive.google.com/drive/folders/1HoKfghXOmiqi028oc_Wv0m2IlLdcJglQ?usp=share_link). You can use [gdown] to download all the pretrained models via command line as follows.

```sh
gdown --id 1HoKfghXOmiqi028oc_Wv0m2IlLdcJglQ --folder
```
