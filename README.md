# minMusicTransformer
Transfer Learning with a Transformer based on https://github.com/salu133445/mmt a multitrack music transformer that generates music given a seed. 
Two different versions of the model are presented and trained in this repository.
1) Retraining of top layers on the MCMA - Multitrack Contrapuntal Music Archive.
   It worked well, but the output is similar to that of the original model.
2) Remodelling to setting counterpoint to a melody,  instead of generating music given a seed - (continuing a sequence). 
   We follow the approach by [Nichols et al](https://arxiv.org/abs/2006.14221).
   This did not work well, perhaps more low-level retraining is required, or maybe the tasks of sequence prediction and sequence translation are too far apart.

Samples with audio and sheet music are available [here](https://www.youtube.com/watch?v=XR9NxTAaMv4)

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

## Required Packages
We recommend using Conda. You can create the environment with the following command.
```sh
conda env create -f environment.yml
```

## Preprocessing


#### Step 1 -- Download the datasets

1) [MCMA dataset](https://mcma.readthedocs.io/en/latest/docs/download.html) - This is what we used for the simple transfer learning, contains the scores (music XML) of 470 contrapuntal pieces from the composers Albinoni, Bach, Becker, Buxtehude, and Lully.
2) [Sequence Translation Dataset](https://gitlab.com/skalo/baroque-nmt/-/tree/master/data?ref_type=heads) - consists of the scores of roughly 700 baroque 2 and 3-voiced pieces and an additional 500 multitrack pieces including orchestral works. The pieces are reconfigured into two track pairs, arbitrarily spliced into 4-measure sections, and filtered to contain more than 10 notes, yielding a dataset of 41,297 four-bar, two-voice segments. 
3)The original model is trained on the  [Symbolic orchestral database (SOD)](https://qsdfo.github.io/LOP/database.html)

Other large data-sets to consider:
- [Lakh MIDI Dataset (LMD)](https://qsdfo.github.io/LOP/database.html):
- [SymphonyNet Dataset](https://symphonynet.github.io/):

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
