import argparse
import os
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

    # tpu
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
    )
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices("TPU"))
    strategy = tf.distribute.TPUStrategy(resolver)

    # config
    config = Config.from_dict(config_dict)
    config.add_git_info()

    output.mkdir(parents=True)
    with (output / "config.yaml").open(mode="w") as f:
        yaml.safe_dump(config.to_dict(), f)

    # dataset
    with tf.device("/job:localhost"):
        datasets = create_dataset(config.dataset, on_memory=True)
        train_dataset = (
            datasets["train"]
            .shuffle(buffer_size=10000)
            .repeat()
            .batch(config.train.batchsize)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        test_dataset = (
            datasets["test"]
            .repeat(config.train.batchsize // config.dataset.num_test)
            .batch(config.train.batchsize)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    # model
    with strategy.scope():
        predictor = create_predictor(
            config.network, quantizer_ema_decay=config.train.quantizer_ema_decay
        )
        model = Model(
            config=config.model,
            predictor=predictor,
            replica_size=strategy.num_replicas_in_sync,
        )

    # optimizer
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(**config.train.optimizer)

    # train
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dataset = strategy.experimental_distribute_dataset(test_dataset)

    model.compile(
        optimizer=optimizer, experimental_steps_per_execution=1,
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(output.joinpath("predictor_{epoch}.hdf5")),
        save_weights_only=True,
        save_freq=config.train.snapshot_iteration,
        verbose=1,
        options=tf.train.CheckpointOptions(experimental_io_device="/job:localhost"),
    )

    terminate_nan_callback = tf.keras.callbacks.TerminateOnNaN()

    history = model.fit(
        train_dataset,
        steps_per_epoch=config.train.log_iteration,
        epochs=config.train.stop_iteration // config.train.log_iteration,
        callbacks=[checkpoint_callback, terminate_nan_callback],
        shuffle=False,
        validation_data=test_dataset,
        validation_freq=1,
        validation_steps=1,
    )

    with tf.device("/job:localhost"):
        writer = tf.summary.create_file_writer(str(output))
        with writer.as_default():
            for i_step in range(
                config.train.stop_iteration // config.train.log_iteration
            ):
                for name, values in history.history.items():
                    tf.summary.scalar(
                        name,
                        values[i_step],
                        step=(1 + i_step) * config.train.log_iteration,
                    )
        writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_yaml_path", type=Path)
    parser.add_argument("output", type=Path)
    train(**vars(parser.parse_args()))
