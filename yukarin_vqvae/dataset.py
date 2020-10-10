import json
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import numpy
import tensorflow as tf
import tensorflow_io as tfio
import yaml
from tqdm import tqdm

from yukarin_vqvae.config import DatasetConfig


def load_audio(path, sampling_rate: int):
    audio = tfio.audio.AudioIOTensor(path, dtype=tf.int16)
    rate = audio.rate
    tf.assert_equal(rate, sampling_rate)
    wave = audio.to_tensor()
    wave = tf.cast(wave, tf.float32) / (2 ** 16)
    return wave


def encode_mulaw(x, num: int):
    mu = tf.constant(num - 1, dtype=x.dtype)
    y = tf.sign(x) * tf.math.log1p(mu * tf.abs(x)) / tf.math.log1p(mu)
    y = tf.cast((y + 1) / 2 * mu + 0.5, tf.int32)
    return y


def encode_mulaw_wrapper(x, num: int):
    y = encode_mulaw(x, num=num)
    y = tf.squeeze(y, -1)
    return dict(wave=x, target=y)


def load_numpy_file(path):
    item = numpy.load(path.numpy(), allow_pickle=True).item()
    return item["array"].squeeze(), item["rate"]


def load_numpy_file_wrapper(path):
    return tf.py_function(load_numpy_file, inp=[path], Tout=(tf.bool, tf.float32))


def process_silence(
    silence, before_rate, after_rate: int, sampling_length: int, min_sound_length: int
):
    scale = int(after_rate // before_rate)
    sound = tf.repeat(~silence, scale)
    sound_point = (
        tf.squeeze(
            tf.nn.conv1d(
                tf.cast(sound, tf.float32)[tf.newaxis, :, tf.newaxis],
                tf.ones([sampling_length + 1, 1, 1]),
                stride=1,
                padding="VALID",
            )
        )
        >= min_sound_length
    )
    return sound_point


def serialize(wave, target, sound_point, speaker_id):
    feature = {
        "wave": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(wave).numpy()])
        ),
        "target": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(target).numpy()]
            )
        ),
        "sound_point": tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[tf.io.serialize_tensor(sound_point).numpy()]
            )
        ),
        "speaker_id": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[speaker_id.numpy()])
        ),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def serialize_wrapper(wave_target: Dict[str, Any], sound_point, speaker_id):
    serialized = tf.py_function(
        serialize,
        inp=[wave_target["wave"], wave_target["target"], sound_point, speaker_id],
        Tout=(tf.string),
    )
    return tf.reshape(serialized, ())


def deserialize(serialized):
    example = tf.io.parse_single_example(
        serialized,
        {
            "wave": tf.io.FixedLenFeature([], dtype=tf.string),
            "target": tf.io.FixedLenFeature([], dtype=tf.string),
            "sound_point": tf.io.FixedLenFeature([], dtype=tf.string),
            "speaker_id": tf.io.FixedLenFeature([], dtype=tf.int64),
        },
    )
    example["wave"] = tf.io.parse_tensor(example["wave"], tf.float32)
    example["target"] = tf.io.parse_tensor(example["target"], tf.int32)
    example["sound_point"] = tf.io.parse_tensor(example["sound_point"], tf.bool)
    return example


def random_offset(data: Dict[str, Any], sampling_length: int):
    wave = data["wave"]
    target = data["target"]
    sound_point = data["sound_point"]
    speaker_id = data["speaker_id"]

    length = tf.shape(wave)[0]
    offset = tf.squeeze(
        tf.random.categorical(
            [
                tf.math.log(
                    tf.cast(sound_point[: length - (sampling_length + 1)], tf.float32)
                )
            ],
            num_samples=1,
        )
    )
    return dict(wave=wave, target=target, offset=offset, speaker_id=speaker_id)


def sample_data(data: Dict[str, Any], sampling_length: int):
    offset = data["offset"]

    wave = data["wave"][offset : offset + sampling_length + 1]
    wave.set_shape([sampling_length + 1, 1])

    target = data["target"][offset + 1 : offset + sampling_length + 1]
    target.set_shape([sampling_length])

    speaker_id = tf.cast(data["speaker_id"], dtype=tf.int32)
    speaker_id = tf.reshape(speaker_id, shape=[1])
    return dict(wave=wave, target=target, speaker_id=speaker_id)


def create_record(
    sampling_rate: int,
    sampling_length: int,
    bin_size: int,
    wave_glob: str,
    silence_glob: str,
    min_sound_length: int,
    speaker_dict_path: str,
    speaker_size: int,
    output_directory: str,
    filename_format: str,
    samples_par_file: int,
    seed: int = None,
    processes: int = None,
):
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(
        locals(), Path(output_directory).joinpath("arguments.yaml").open(mode="w")
    )

    if processes is None:
        processes = tf.data.experimental.AUTOTUNE

    wave_paths = {Path(p).stem: Path(p) for p in glob(str(wave_glob))}
    names = sorted(wave_paths.keys())
    assert len(names) > 0

    silence_paths = {Path(p).stem: Path(p) for p in glob(str(silence_glob))}
    assert set(names) == set(silence_paths.keys())

    name_each_speaker: Dict[str, List[str]] = json.load(open(speaker_dict_path))
    assert speaker_size == len(name_each_speaker)

    speaker_ids = {
        name: speaker_id
        for speaker_id, (_, ns) in enumerate(name_each_speaker.items())
        for name in ns
    }
    assert set(names).issubset(set(speaker_ids.keys()))

    numpy.random.RandomState(seed).shuffle(names)

    mulaw_dataset = (
        tf.data.Dataset.from_tensor_slices([str(wave_paths[name]) for name in names])
        .map(partial(load_audio, sampling_rate=sampling_rate))
        .map(
            partial(encode_mulaw_wrapper, num=bin_size),
            num_parallel_calls=processes,
        )
    )

    sound_dataset = (
        tf.data.Dataset.from_tensor_slices([str(silence_paths[name]) for name in names])
        .map(load_numpy_file_wrapper, num_parallel_calls=processes)
        .map(
            partial(
                process_silence,
                after_rate=sampling_rate,
                sampling_length=sampling_length,
                min_sound_length=min_sound_length,
            ),
            num_parallel_calls=processes,
        )
    )

    speaker_dataset = tf.data.Dataset.from_tensor_slices(
        [speaker_ids[name] for name in names]
    )

    array_dataset = tf.data.Dataset.zip((mulaw_dataset, sound_dataset, speaker_dataset))
    serialized_dataset = array_dataset.map(serialize_wrapper)

    for i_file in tqdm(
        range(int(numpy.ceil(len(names) / samples_par_file))), desc="create_record"
    ):
        i_data = i_file * samples_par_file
        filename = str(filename_format).format(i_data=i_data, i_file=i_file)
        writer = tf.data.experimental.TFRecordWriter(
            str(Path(output_directory).joinpath(filename)),
            compression_type="GZIP",
        )
        writer.write(
            serialized_dataset.skip(i_data)
            .take(samples_par_file)
            .prefetch(samples_par_file)
        )


def create_dataset(config: DatasetConfig, on_memory: bool = False):
    record_dataset = tf.data.TFRecordDataset(
        glob(str(config.dataset_glob)),
        compression_type="GZIP",
        num_parallel_reads=tf.data.experimental.AUTOTUNE,
    )

    if on_memory:
        record_dataset = tf.data.Dataset.from_tensor_slices(list(iter(record_dataset)))

    dataset = (
        record_dataset.map(deserialize)
        .cache()
        .map(
            partial(random_offset, sampling_length=config.sampling_length),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        .map(
            partial(sample_data, sampling_length=config.sampling_length),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
    )

    num_test = config.num_test
    if config.num_train is None:
        return dict(test=dataset.take(num_test), train=dataset.skip(num_test))
    else:
        return dict(
            test=dataset.take(num_test),
            train=dataset.skip(num_test).take(config.num_train),
        )
