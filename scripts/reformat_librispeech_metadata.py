import pandas as pd
import json
import os
import argparse
import textgrid
from tqdm import tqdm

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_al_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--librispeech_md_dir', type=str, required=True,
                    help='Path to librispeech metadata directory')


def main(args):
    librispeech_al_dir = args.librispeech_al_dir
    librispeech_md_dir = args.librispeech_md_dir
    out_dir = 'metadata/reformated_LibriSpeech'
    os.makedirs('metadata/reformated_LibriSpeech', exist_ok=True)
    for file in os.listdir(librispeech_md_dir):
        md = pd.read_csv(os.path.join(librispeech_md_dir, file))
        new_file = process(md, librispeech_al_dir)
        save_path = os.path.join(out_dir, file).replace('.csv', '.json')
        with open(save_path, 'w') as f:
            json.dump(new_file, f, indent=4)


def process(file, librispeech_al_dir):
    new_file = []

    for row in tqdm(file.iterrows()):
        row = row[1]
        temp_dict = {'speaker_ID': row['speaker_ID'], 'sex': row['sex'], 'subset': row['subset'],
                     'length': row['length'], 'origin_path': row['origin_path'],
                     'VAD': {'start': [], 'stop': [], 'word': []}}
        try:
            tg = textgrid.TextGrid.fromFile(
                os.path.join(librispeech_al_dir, row['origin_path'].replace('.flac', '.TextGrid')))
            for interval in tg[0]:
                if interval.mark != '':
                    temp_dict['VAD']['start'].append(interval.minTime)
                    temp_dict['VAD']['stop'].append(interval.maxTime)
                    temp_dict['VAD']['word'].append(interval.mark)
        except FileNotFoundError:
            pass
        new_file.append(temp_dict)
    return new_file


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
