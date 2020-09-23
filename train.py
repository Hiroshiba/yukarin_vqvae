import argparse
from pathlib import Path

import tensorflow as tf
import yaml

from yukarin_vqvae.config import Config
from yukarin_vqvae.dataset import create_dataset
from yukarin_vqvae.model import Model
from yukarin_vqvae.networks.predictor import create_predictor


def train(
    config_yaml_path: Path, output: Path,
):
    with config_yaml_path.open() as f:
        config_dict = yaml.safe_load(f)

    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # model
    predictor = create_predictor(
        config.network, quantizer_ema_decay=config.train.quantizer_ema_decay
    )
    model = Model(config=config.model, predictor=predictor, replica_size=1,)

    # dataset
    datasets = create_dataset(config.dataset)
    train_dataset = (
        datasets["train"]
        .shuffle(buffer_size=10000)
        .repeat()
        .batch(config.train.batchsize)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    test_dataset = (
        datasets["test"]
        .batch(config.train.batchsize)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    # optimizer
    optimizer = tf.keras.optimizers.Adam(**config.train.optimizer)

    # worm up
    writer = tf.summary.create_file_writer(str(output))

    @tf.function
    def wrapper():
        return model(next(iter(train_dataset)), training=True)

    tf.summary.trace_on(graph=True)
    wrapper()
    with writer.as_default():
        tf.summary.trace_export(name="graph", step=0)

    # train
    model.compile(optimizer=optimizer)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(output), profile_batch=0,
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output.joinpath("model_{epoch}.hdf5")),
        save_weights_only=True,
        save_freq=config.train.snapshot_iteration,
        verbose=1,
    )

    terminate_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    model.fit(
        train_dataset,
        steps_per_epoch=config.train.log_iteration,
        epochs=config.train.stop_iteration // config.train.log_iteration,
        callbacks=[tensorboard_callback, checkpoint_callback, terminate_nan_callback],
        shuffle=False,
        validation_data=test_dataset,
        validation_freq=1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output", type=Path)
    train(**vars(parser.parse_args()))
