# -*- coding: utf-8 -*-
# Copyright 2021 The Dapr Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import base64
import sys
import time
from pathlib import Path

import msgpack
import msgpack_numpy as m
import numpy as np
from dapr.actor import ActorId, ActorProxy
from mnist_actor_interface import MnistActorInterface

np.random.seed(42)


async def main():
    root_folder = Path(__file__).parents[0] / "./datasets"

    assert root_folder.is_dir(), f"Wrong folder: {root_folder}"

    TRAINING_SET_IMAGES = root_folder / "train-images-idx3-ubyte"
    TRAINING_SET_LABELS = root_folder / "train-labels-idx1-ubyte"

    TEST_SET_IMAGES = root_folder / "t10k-images-idx3-ubyte"
    TEST_SET_LABELS = root_folder / "t10k-labels-idx1-ubyte"

    def fetch(file: Path):
        data = b""
        with file.open("rb") as fh:
            data = fh.read()
        return np.frombuffer(data, dtype=np.uint8).copy()

    SKIP_HEXADECIMAL_16 = 0x10
    SKIP_DECIMAL_8 = 8
    # 3D array: (x, y, z)
    # Shape is the number of cells
    # Shape: (47_040_016,); Size: 47_040_016
    # Shape: (47_040_000,); Size: 47_040_000
    # Shape: (60_000, 28, 28); Size: 47_040_000
    # Resolution 28 px width x 28 px height
    X = fetch(TRAINING_SET_IMAGES)[SKIP_HEXADECIMAL_16:].reshape((-1, 28, 28))
    # Shape: (60_008,); Size: 60_008
    # Shape: (60_000,); Size: 60_000
    Y = fetch(TRAINING_SET_LABELS)[SKIP_DECIMAL_8:]
    # Shape: (10_000, 784); Size: 7840000
    X_test = fetch(TEST_SET_IMAGES)[SKIP_HEXADECIMAL_16:].reshape((-1, 28 * 28))
    # Shape: (10_000,); Size: 10_000
    Y_test = fetch(TEST_SET_LABELS)[SKIP_DECIMAL_8:]

    # Validation split
    rand = np.arange(60_000)
    np.random.shuffle(rand)

    train_indices = rand[:50_000]
    validator_indices = np.setdiff1d(rand, train_indices)

    X_train = X[train_indices, :, :]
    X_validation = X[validator_indices, :, :]

    Y_train = Y[train_indices]
    Y_validation = Y[validator_indices]
    LEARNING_RATE = 0.001

    # Create proxy client
    mnist_proxy = ActorProxy.create("MnistActor", ActorId("1"), MnistActorInterface)

    print("MnistActor.set_up", flush=True)
    await mnist_proxy.SetUp(LEARNING_RATE)

    # Training
    EPOCHS = 50
    TRAINING_BATCH_SIZE = 128
    i = 1
    for i in range(1, EPOCHS + 1):
        print(f"Epoch %: {(i / EPOCHS) * 100:.2f}", flush=True)
        print("BankActor.run", flush=True)
        rtn_obj = await mnist_proxy.Run()
        print(f"Return object: {rtn_obj}", flush=True)
        # Randomize and create batches
        # sample = np.random.randint(
        #     0, X_train.shape[0], size=TRAINING_BATCH_SIZE
        # )  # 128 random samples
        sample = np.random.randint(
            0, X_train.shape[0], size=X_train.shape[0]
        )  # 60_000 random samples
        max_batch = X_train.shape[0] // TRAINING_BATCH_SIZE - 1
        for batch in range(max_batch):
            batch_percent = ((batch + 1) / max_batch) * 100
            print(f"Batch %: {batch_percent:.2f}", flush=True)
            start = batch * TRAINING_BATCH_SIZE
            end = (batch + 1) * TRAINING_BATCH_SIZE
            partial_sample = sample[start:end]
            x: np.ndarray = X_train[partial_sample].reshape((-1, 28 * 28))
            y: np.ndarray = Y_train[partial_sample]

            # print(f"x {x.size * x.itemsize:d} bytes", flush=True)
            # print(f"y {y.size * y.itemsize:d} bytes", flush=True)
            data_raw = {
                "x_train": x,
                "y_train": y,
            }
            data_enc = {
                "x_train": msgpack.packb(data_raw["x_train"], default=m.encode),
                "y_train": msgpack.packb(data_raw["y_train"], default=m.encode),
            }
            print("MnistActor.forward_backward_pass", flush=True)
            await mnist_proxy.ForwardBackwardPass(data_enc)

        # Validating our model every epoch
        data_raw = {
            "X_validation": X_validation.reshape((-1, 28 * 28)),
            "Y_validation": Y_validation,
        }
        data_enc = {
            "X_validation": msgpack.packb(data_raw["X_validation"], default=m.encode),
            "Y_validation": msgpack.packb(data_raw["Y_validation"], default=m.encode),
        }
        print("MnistActor.validate", flush=True)
        await mnist_proxy.Validate(data_enc)

        print("MnistActor.backup_mnist_data", flush=True)
        await mnist_proxy.BackupMnistData(i)

        print("BankActor.finish", flush=True)
        rtn_obj = await mnist_proxy.Finish()
        print(f"Return object: {rtn_obj}", flush=True)
        total = rtn_obj["end"] - rtn_obj["start"]
        ITERATIONS = TRAINING_BATCH_SIZE
        print(f"Did {ITERATIONS} transactions in {total} s", flush=True)
        print(f"{ITERATIONS / total} transactions per second", flush=True)

    time.sleep(1)
    rtn_obj = {}
    try:
        print("MnistActor.get_mnist_data", flush=True)
        rtn_obj = await mnist_proxy.GetMnistData(i)

        rtn_obj["l1_regularization"] = base64.b64decode(rtn_obj["l1_regularization"])
        rtn_obj["l1_regularization"] = msgpack.unpackb(
            rtn_obj["l1_regularization"], object_hook=m.decode
        )

        rtn_obj["l2_regularization"] = base64.b64decode(rtn_obj["l2_regularization"])
        rtn_obj["l2_regularization"] = msgpack.unpackb(
            rtn_obj["l2_regularization"], object_hook=m.decode
        )
    except Exception as e:
        print(f"MnistActor.get_mnist_data error: {e}", flush=True)
        exception_type, exception_object, exception_traceback = sys.exc_info()
        filename = exception_traceback.tb_frame.f_code.co_filename
        line_number = exception_traceback.tb_lineno

        print(f"Exception type: {exception_type}", flush=True)
        print(f"File name: {filename}", flush=True)
        print(f"Line number: {line_number}", flush=True)
    print(f"Get MNIST data: {rtn_obj}", flush=True)

    np.savez_compressed(
        "weights_compressed",
        l1=rtn_obj["l1_regularization"],
        l2=rtn_obj["l2_regularization"],
    )  # "weights_compressed.npz"

    def sigmoid(x_val: np.ndarray):
        """Sigmoid.

        Attributes:
            x_val: Shape: (128, 128).
        """
        result = 1 / (np.exp(-x_val) + 1)  # Shape: (128, 128)
        return result

    def read_numpy():
        compressed_file = Path(__file__).parents[0] / "./weights_compressed.npz"
        assert compressed_file.is_file(), f"Wrong file: {compressed_file}"
        data = np.load(str(compressed_file), allow_pickle=True)

        print("L1 regularization", flush=True)
        print(data["l1"], flush=True)
        print("L2 regularization", flush=True)
        print(data["l2"], flush=True)

        # Number 7
        m = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 10, 10, 10, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 10, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]

        m = np.concatenate([np.concatenate([[x] * 4 for x in y] * 4) for y in m])
        m = m.reshape(1, -1)
        step_1 = m.dot(data["l1"])
        step_2 = sigmoid(step_1).dot(data["l2"])
        x = np.argmax(step_2, axis=1)
        print(f"First attempt: Expected 7; Actual: {x}", flush=True)

        # Number 1
        n = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]

        n = np.concatenate([np.concatenate([[x] * 4 for x in y] * 4) for y in n])
        n = n.reshape(1, -1)
        step_1 = n.dot(data["l1"])
        step_2 = sigmoid(step_1).dot(data["l2"])
        x = np.argmax(step_2, axis=1)
        print(f"Second attempt: Expected 1; Actual: {x}", flush=True)

    print("Validate trained data against two known numbers", flush=True)
    read_numpy()

    time.sleep(10)


asyncio.run(main())
