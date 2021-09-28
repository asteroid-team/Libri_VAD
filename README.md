#About the dataset

LibriVAD is an open source dataset for voice activity detection in noisy environments. 
It is derived from LibriSpeech signals (clean subset) and DNS challenge noises.

#Generating LibriVAD

You need to download [LibriSpeech](https://www.openslr.org/12), the noise from the [DNS Challenge](https://github.com/microsoft/DNS-Challenge) (datasets/noise)
and the [forced alignments](https://zenodo.org/record/2619474#.YVLPu3s6_JV).

To generate LibriVAD, clone the repo and run the main script : `run.sh`
(edit `run.sh` with correct paths)

```
git clone https://github.com/JorisCos/LibriMix
cd LibriMix 
./run.sh storage_dir
```
