import argparse
from pathlib import Path

from yukarin_vqvae.dataset import create_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling_rate", required=True, type=int)
    parser.add_argument("--sampling_length", required=True, type=int)
    parser.add_argument("--bin_size", required=True, type=int)
    parser.add_argument("--wave_glob", required=True, type=str)
    parser.add_argument("--silence_glob", required=True, type=str)
    parser.add_argument("--min_sound_length", required=True, type=int)
    parser.add_argument("--speaker_dict_path", required=True, type=str)
    parser.add_argument("--speaker_size", required=True, type=int)
    parser.add_argument("--output_directory", required=True, type=str)
    parser.add_argument("--filename_format", required=True, type=str)
    parser.add_argument("--samples_par_file", required=True, type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--processes", type=int)
    create_record(**vars(parser.parse_args()))
