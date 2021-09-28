#!/bin/bash
set -eu  # Exit on error

storage_dir=$1
source_dir=$storage_dir/LibriSpeech
noise_dir=$storage_dir/DNS_noise
outdir=$storage_dir/VAD_dataset

python scripts/reformat_librispeech_metadata.py --librispeech_al_dir $storage_dir/librispeech_alignments --librispeech_md_dir metadata/LibriSpeech
python scripts/create_noise_metadata.py --dns_dir $noise_dir
python scripts/create_VAD_dataset.py --librispeech_dir $source_dir --metadata_dir metadata/ --outdir $outdir
