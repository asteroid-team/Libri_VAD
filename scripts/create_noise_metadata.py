import os
import random

import soundfile as sf
from tqdm import tqdm
import numpy as np
import librosa
import json
import argparse

target_sr = 16000

parser = argparse.ArgumentParser()
parser.add_argument('--dns_dir', type=str, required=True,
                    help='Path to DNS_noise root directory')
np.random.seed(21)


def main(args):
    noise_dir = args.dns_dir
    train_file = []
    dev_file = []
    test_file = []
    _index = random.sample([i for i in range(65000)], 4534)
    test_index = _index[:2230]
    dev_index = _index[2230:]
    with os.scandir(noise_dir) as it:
        i = 0
        for entry in tqdm(it):
            file_path = os.path.join(noise_dir, entry)
            file = sf.SoundFile(file_path)
            if i in dev_index:
                dev_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                 '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
            elif i in test_index:
                test_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                  '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
            else:
                train_file.append({'path': file_path, 'duration': len(file), 'sample_rate': file.samplerate,
                                   '16kHz_duration': int(len(file) / file.samplerate * target_sr)})
            i += 1

    os.makedirs('metadata/dns_noise', exist_ok=True)
    with open(os.path.join('metadata/dns_noise', 'train.json'), 'w') as outfile:
        json.dump(train_file, outfile, indent=4)
    with open(os.path.join('metadata/dns_noise', 'dev.json'), 'w') as outfile:
        json.dump(dev_file, outfile, indent=4)
    with open(os.path.join('metadata/dns_noise', 'test.json'), 'w') as outfile:
        json.dump(test_file, outfile, indent=4)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
