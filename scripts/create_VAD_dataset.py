from librosa import resample
import tqdm.contrib.concurrent
import argparse
import os
import json
import numpy as np
import numpy.random as npr
import soundfile as sf
import random
import warnings
import pyloudnorm as pyln
from tqdm import tqdm
from itertools import cycle

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000
TARGET_SR = 8000
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25
MAX_AMP = 0.9
np.random.seed(22)

parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--outdir', type=str, default=None,
                    help='Path to the desired dataset root directory')


def main(args):
    # Get librispeech root path
    global librispeech_dir
    librispeech_dir = args.librispeech_dir
    # Get Metadata directory
    metadata_dir = args.metadata_dir
    # Get LibriMix root path
    vad_outdir = args.outdir

    noise_md = sorted(os.listdir(os.path.join(metadata_dir, 'dns_noise')))
    source_md = sorted(os.listdir(os.path.join(metadata_dir, 'reformated_LibriSpeech')))
    source_md = ['train-clean-360.json']
    noise_md = ['train.json']
    # source_md.remove('train-clean-100.json')

    # Get the desired frequencies
    for source_file, noise_file in zip(source_md, cycle(noise_md)):
        with open(os.path.join(metadata_dir, 'dns_noise', noise_file)) as file:
            n_file = json.load(file)
        with open(os.path.join(metadata_dir, 'reformated_LibriSpeech', source_file)) as file:
            s_file = json.load(file)
        dir_name = noise_file.replace('.json', '')
        dir_path = os.path.join(vad_outdir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
        process(s_file, n_file, metadata_dir, dir_path)


def process(source_file, noise_file, metadata_dir, dir_path):
    result_md = []

    for s_file, n_file in tqdm(zip(source_file, noise_file), total=len(source_file)):
        max_swap = len(s_file['VAD']['start'])
        n_swap = draw_n_swap(max_swap)
        cuts = fusion_VAD(s_file[f'VAD'], n_swap)

        cuts = {'start': [int(cuts['start'][j] * TARGET_SR) for j in range(len(cuts['start']))],
                'stop': [int(cuts['stop'][j] * TARGET_SR) for j in range(len(cuts['start']))],
                'word': cuts['word']}

        source = read_sources(s_file)
        source, VAD = align_sources(source, cuts)
        noise = read_noise(n_file)
        noise = fit_noise(noise, len(source))
        sources = (source + [noise])
        if len(source) < TARGET_SR * 3:
            print('too short')
            continue
        # sources = pad(sources)
        loudness_list, _, source_norm_list = set_loudness(sources)
        mixture = mix(source_norm_list)
        m_length = len(mixture)
        create_directories(dir_path)
        mixture_path = write_sources_and_mixture(mixture, s_file, dir_path)
        add_to_md(result_md, mixture_path, n_swap, VAD, m_length)

    if 'train' in dir_path:
        save_md_path = 'train.json'
    elif 'test' in dir_path:
        save_md_path = 'test.json'
    else:
        save_md_path = 'dev.json'
    os.makedirs((os.path.join(metadata_dir, 'sets')), exist_ok=True)
    with open(os.path.join(metadata_dir, 'sets', save_md_path),
              'w') as outfile:
        json.dump(result_md, outfile, indent=4)


def draw_n_swap(max_swap):
    # global overlap over the whole mixture
    return int(np.random.default_rng().uniform(1, max_swap, 1)[0])


def fusion_VAD(VAD_dict, n_swap):
    while len(VAD_dict['start']) != n_swap:
        to_fusion = int(np.random.default_rng().uniform(low=1, high=len(VAD_dict['start'])))
        del (VAD_dict['start'][to_fusion])
        del (VAD_dict['stop'][to_fusion - 1])
    return VAD_dict


def read_sources(mixture):
    source = sf.read(os.path.join(librispeech_dir, mixture[f'origin_path']), dtype='float32')[0]
    source = resample(source, RATE, TARGET_SR)
    return source


def read_noise(mixture):
    noise, n_sr = sf.read(mixture['path'], dtype='float32')
    noise = resample(noise, n_sr, TARGET_SR)
    return noise


def fit_noise(noise, duration):
    if len(noise) > duration:
        noise = noise[:duration]
    elif len(noise) < duration:
        noise = extend_noise(noise, duration)
    return noise


def extend_noise(noise, max_length):
    """ Concatenate noise using hanning window"""
    noise_ex = noise
    window = np.hanning(RATE // 4 + 1)
    # Increasing window
    i_w = window[:len(window) // 2 + 1]
    # Decreasing window
    d_w = window[len(window) // 2::-1]
    # Extend until max_length is reached
    while len(noise_ex) < max_length:
        noise_ex = np.concatenate((noise_ex[:len(noise_ex) - len(d_w)],
                                   np.multiply(
                                       noise_ex[len(noise_ex) - len(d_w):],
                                       d_w) + np.multiply(
                                       noise[:len(i_w)], i_w),
                                   noise[len(i_w):]))
    noise_ex = noise_ex[:max_length]
    return noise_ex


def align_sources(sources, cuts):
    source_aligned = np.array([])
    VAD = {'start': [], 'stop': []}
    VAD['start'].append(0)
    for i in range(len(cuts['start'])):
        chunks = get_chunks(sources, cuts, i)
        # chunks = smooth(chunks)
        source_aligned = np.concatenate((source_aligned, chunks))
        VAD['stop'].append(len(source_aligned))
        silence_length = random.randrange(int(TARGET_SR * 0.2), TARGET_SR)
        silence = np.zeros(silence_length)
        source_aligned = np.concatenate((source_aligned, silence))
        VAD['start'].append(len(source_aligned))
    del (VAD['start'][-1])
    return source_aligned, VAD


def get_chunks(source, cuts, i):
    return source[cuts['start'][i]:cuts['stop'][i]]


def set_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(TARGET_SR)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS - 5,
                                             MAX_LOUDNESS - 5)
        # Normalize source to target loudness
        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max


def add_to_md(result_md, mixture_path, n_swap, VAD, m_length):
    row = {'mixture_path': mixture_path, 'n_swap': n_swap,
           'VAD': VAD, 'length': m_length}
    result_md.append(row)


def write_sources_and_mixture(mixture, file, dir_path):
    name = (os.path.basename(file[f'origin_path'])).replace('.flac', '.wav')
    mixture_path = os.path.join(dir_path, 'mixture', name)
    sf.write(mixture_path, mixture, TARGET_SR)
    return mixture_path


def create_directories(path):
    os.makedirs(os.path.join(path, 'mixture'), exist_ok=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
