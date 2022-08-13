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

import base64
import datetime
import logging
import sys
import time
from typing import Any, List, Optional, TypedDict

import msgpack
import msgpack_numpy as m
import numpy as np
from dapr.actor import Actor, Remindable
from mnist_actor_interface import MnistActorInterface

logger = logging.getLogger("MnistActor")


class TimerData(TypedDict):
    ts: datetime.datetime
    start: float
    end: float

class MnistState(TypedDict):
    learning_rate: float
    l1_regularization: Optional[Any]
    l2_regularization: Optional[Any]
    update_l1_regularization: Optional[Any]
    update_l2_regularization: Optional[Any]
    output: Optional[Any]


class MnistTrainingState(TypedDict):
    losses: List
    accuracies: List


class MnistValidationState(TypedDict):
    validation_accuracies: List


class MnistActor(Actor, MnistActorInterface, Remindable):
    def __init__(self, ctx, actor_id):
        super(MnistActor, self).__init__(ctx, actor_id)

    async def _on_activate(self) -> None:
        """A callback which will be called whenever actor is activated."""
        print(f"Activate {self.__class__.__name__} actor!", flush=True)

    async def _on_deactivate(self) -> None:
        """A callback which will be called whenever actor is deactivated."""
        print(f"Deactivate {self.__class__.__name__} actor!", flush=True)

    async def receive_reminder(
        self,
        name: str,
        state: bytes,
        due_time: datetime.timedelta,
        period: datetime.timedelta,
        ttl: Optional[datetime.timedelta] = None,
    ) -> None:
        """A callback which will be called when reminder is triggered."""
        print(
            f"receive_reminder is called - {name} reminder - {str(state)}", flush=True
        )

    async def set_up(self, learning_rate: float) -> None:
        """An actor method to set up class."""
        print(f"MnistActor.set_up", flush=True)
        # mnist_timer_state
        data_raw: TimerData = {
            "ts": datetime.datetime.now(datetime.timezone.utc),
            "start": -1,
            "end": -1,
        }
        await self._state_manager.set_state("mnist_timer_state", data_raw)
        await self._state_manager.save_state()
        # mnist_state
        data_raw: MnistState = {
            "learning_rate": learning_rate,
            "l1_regularization": await self._initialize_weight(28 * 28, 128),
            "l2_regularization": await self._initialize_weight(128, 10),
            "update_l1_regularization": None,
            "update_l2_regularization": None,
            "output": None,
        }
        data_enc = {
            **data_raw,
            **{
                "l1_regularization": msgpack.packb(
                    data_raw["l1_regularization"], default=m.encode
                ),
                "l2_regularization": msgpack.packb(
                    data_raw["l2_regularization"], default=m.encode
                ),
            },
        }
        await self._state_manager.set_state("mnist_state", data_enc)
        await self._state_manager.save_state()
        # mnist_training_state
        data_raw: MnistTrainingState = {
            "losses": [],
            "accuracies": [],
        }
        data_enc = {**data_raw}
        await self._state_manager.set_state("mnist_training_state", data_enc)
        await self._state_manager.save_state()
        # mnist_validation_state
        data_raw: MnistValidationState = {
            "validation_accuracies": [],
        }
        data_enc = {**data_raw}
        await self._state_manager.set_state("mnist_validation_state", data_enc)
        await self._state_manager.save_state()

    async def backup_mnist_data(self, iteration: int) -> None:
        """Backup MNIST data."""
        print(f"MnistActor.backup_mnist_data", flush=True)
        has_value, val = await self._state_manager.try_get_state("mnist_state")
        print(f"Iteration {iteration}", flush=True)

        try:
            val["iteration"] = iteration
            await self._state_manager.set_state(f"mnist_state_backup_{iteration}", val)
            await self._state_manager.save_state()
        except Exception as e:
            print(f"MnistActor.backup_mnist_data error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    @staticmethod
    async def _initialize_weight(x_val: int, y_val: int) -> np.ndarray:
        """Initialize weights."""
        layer = np.random.uniform(-1.0, 1.0, size=(x_val, y_val)) / np.sqrt(
            x_val * y_val
        )
        result = layer.astype(np.float32)  # Shape: (784, 128)
        return result

    @staticmethod
    async def _sigmoid(x_val: np.ndarray):
        """Sigmoid.

        Attributes:
            x_val: Shape: (128, 128).
        """
        result = 1 / (np.exp(-x_val) + 1)  # Shape: (128, 128)
        return result

    @staticmethod
    async def _sigmoid_derivative(x_val):
        """Sigmoid derivative.

        Attributes:
            x_val: Shape: (128, 128).
        """
        numerator = np.exp(-x_val)
        denominator = (np.exp(-x_val) + 1) ** 2
        result = numerator / denominator  # Shape: (128, 128)
        return result

    @staticmethod
    async def _softmax(x_val):
        """Softmax.

        Attributes:
            x_val: Shape: (128, 10).
        """
        exp_element = np.exp(x_val - x_val.max())
        result = exp_element / np.sum(exp_element, axis=0)  # Shape: (128, 10)
        return result

    @staticmethod
    async def _softmax_derivative(x_val):
        """Softmax derivative.

        Attributes:
            x_val: Shape: (128, 10).
        """
        exp_element = np.exp(x_val - x_val.max())
        result = (
            exp_element
            / np.sum(exp_element, axis=0)
            * (1 - exp_element / np.sum(exp_element, axis=0))
        )  # Shape: (128, 10)
        return result

    async def forward_backward_pass(self, data_enc):
        """Forward backward pass."""
        print(f"MnistActor.forward_backward_pass", flush=True)
        """Forward and backward pass."""
        try:
            data_enc["x_train"] = base64.b64decode(data_enc["x_train"])
            data_enc["y_train"] = base64.b64decode(data_enc["y_train"])

            data_raw = {
                "x_train": msgpack.unpackb(data_enc["x_train"], object_hook=m.decode),
                "y_train": msgpack.unpackb(data_enc["y_train"], object_hook=m.decode),
            }

            targets = np.zeros((len(data_raw["y_train"]), 10), np.float32)
            targets[range(targets.shape[0]), data_raw["y_train"]] = 1

            has_value, val_enc = await self._state_manager.try_get_state("mnist_state")
            # Base64 string to bytes
            val_enc["l1_regularization"] = base64.b64decode(
                val_enc["l1_regularization"]
            )
            val_enc["l2_regularization"] = base64.b64decode(
                val_enc["l2_regularization"]
            )

            val_raw: MnistState = {
                **val_enc,
                "l1_regularization": msgpack.unpackb(
                    val_enc["l1_regularization"], object_hook=m.decode
                ),
                "l2_regularization": msgpack.unpackb(
                    val_enc["l2_regularization"], object_hook=m.decode
                ),
            }
            x_l1p = data_raw["x_train"].dot(val_raw["l1_regularization"])
            x_sigmoid = await self._sigmoid(x_l1p)

            x_l2p = x_sigmoid.dot(val_raw["l2_regularization"])
            val_raw["output"] = await self._softmax(x_l2p)

            error = (
                2
                * (val_raw["output"] - targets)
                / val_raw["output"].shape[0]
                * await self._softmax_derivative(x_l2p)
            )

            val_raw["update_l2_regularization"] = x_sigmoid.T @ error

            error = (
                val_raw["l2_regularization"].dot(error.T)
            ).T * await self._sigmoid_derivative(x_l1p)
            val_raw["update_l1_regularization"] = data_raw["x_train"].T @ error

            val_enc["l1_regularization"] = msgpack.packb(
                val_raw["l1_regularization"], default=m.encode
            )
            val_enc["l2_regularization"] = msgpack.packb(
                val_raw["l2_regularization"], default=m.encode
            )
            val_enc["update_l1_regularization"] = msgpack.packb(
                val_raw["update_l1_regularization"], default=m.encode
            )
            val_enc["update_l2_regularization"] = msgpack.packb(
                val_raw["update_l2_regularization"], default=m.encode
            )
            val_enc["output"] = msgpack.packb(val_raw["output"], default=m.encode)

            await self._state_manager.set_state("mnist_state", val_enc)

            await self.set_training_data(val_raw["output"], data_raw["y_train"])
            await self.stochastic_gradient_descent()
        except Exception as e:
            print(f"MnistActor.forward_backward_pass error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    async def stochastic_gradient_descent(self) -> None:
        """Stochastic gradient descent."""
        print(f"MnistActor.stochastic_gradient_descent", flush=True)
        try:
            has_value, val = await self._state_manager.try_get_state("mnist_state")
            val_raw: MnistState = {
                **val,
                **{
                    "update_l1_regularization": msgpack.unpackb(
                        val["update_l1_regularization"], object_hook=m.decode
                    ),
                    "l1_regularization": msgpack.unpackb(
                        val["l1_regularization"], object_hook=m.decode
                    ),
                    "update_l2_regularization": msgpack.unpackb(
                        val["update_l2_regularization"], object_hook=m.decode
                    ),
                    "l2_regularization": msgpack.unpackb(
                        val["l2_regularization"], object_hook=m.decode
                    ),
                    "output": msgpack.unpackb(val["output"], object_hook=m.decode),
                },
            }

            val_raw["l1_regularization"] = np.subtract(
                val_raw["l1_regularization"],
                (val_raw["learning_rate"] * val_raw["update_l1_regularization"]),
            )

            val_raw["l2_regularization"] = np.subtract(
                val_raw["l2_regularization"],
                (val_raw["learning_rate"] * val_raw["update_l2_regularization"]),
            )

            val["l1_regularization"] = msgpack.packb(
                val_raw["l1_regularization"], default=m.encode
            )
            val["l2_regularization"] = msgpack.packb(
                val_raw["l2_regularization"], default=m.encode
            )

            await self._state_manager.set_state("mnist_state", val)
            await self._state_manager.save_state()
        except Exception as e:
            print(f"MnistActor.stochastic_gradient_descent error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    async def validate(self, data) -> None:
        """Validate."""
        print(f"MnistActor.validate", flush=True)
        try:
            data["X_validation"] = base64.b64decode(data["X_validation"])
            data["Y_validation"] = base64.b64decode(data["Y_validation"])
            data_raw = {
                "X_validation": msgpack.unpackb(
                    data["X_validation"], object_hook=m.decode
                ),
                "Y_validation": msgpack.unpackb(
                    data["Y_validation"], object_hook=m.decode
                ),
            }

            has_value, val = await self._state_manager.try_get_state("mnist_state")
            val["l1_regularization"] = base64.b64decode(val["l1_regularization"])
            l1_regularization = msgpack.unpackb(
                val["l1_regularization"], object_hook=m.decode
            )
            step_1 = data_raw["X_validation"].dot(l1_regularization)
            val["l2_regularization"] = base64.b64decode(val["l2_regularization"])
            l2_regularization = msgpack.unpackb(
                val["l2_regularization"], object_hook=m.decode
            )
            step_2 = (await self._sigmoid(step_1)).dot(l2_regularization)
            step_3 = await self._softmax(step_2)
            val_out = np.argmax(step_3, axis=1)
            validation_accuracy = (val_out == data_raw["Y_validation"]).mean()
            await self.set_validation_data(validation_accuracy)
        except Exception as e:
            print(f"MnistActor.validate error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    async def get_training_data(self) -> object:
        """Get training data."""
        print(f"MnistActor.get_training_data", flush=True)
        has_value, val = await self._state_manager.try_get_state("mnist_training_state")
        return val

    async def set_training_data(self, output, y_val) -> None:
        """Set training data."""
        print(f"MnistActor.set_training_data", flush=True)
        try:
            category: np.ndarray = np.argmax(output, axis=1)
            accuracy = (category == y_val).mean()
            has_value, val = await self._state_manager.try_get_state(
                "mnist_training_state"
            )
            val["accuracies"].append(accuracy.item())  # Cast from float64 to float
            loss = ((category - y_val) ** 2).mean()
            val["losses"].append(loss.item())  # Cast from float64 to float
            print(f"Training accuracy: {accuracy:.3f}; loss {loss:.3f}")
            await self._state_manager.set_state("mnist_training_state", val)
            await self._state_manager.save_state()
        except Exception as e:
            print(f"MnistActor.set_training_data error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    async def get_validation_data(self) -> object:
        """Get validation data."""
        print(f"MnistActor.get_validation_data", flush=True)
        has_value, val = await self._state_manager.try_get_state(
            "mnist_validation_state"
        )
        return val

    async def set_validation_data(self, data) -> None:
        """Set validation data."""
        print(f"MnistActor.set_validation_data", flush=True)
        try:
            print(f"Validation accuracy: {data:.3f}", flush=True)
            has_value, val = await self._state_manager.try_get_state(
                "mnist_validation_state"
            )
            val["validation_accuracies"].append(data.item())
            await self._state_manager.set_state("mnist_validation_state", val)
            await self._state_manager.save_state()
        except Exception as e:
            print(f"MnistActor.set_validation_data error: {e}", flush=True)
            exception_type, exception_object, exception_traceback = sys.exc_info()
            filename = exception_traceback.tb_frame.f_code.co_filename
            line_number = exception_traceback.tb_lineno

            print(f"Exception type: {exception_type}", flush=True)
            print(f"File name: {filename}", flush=True)
            print(f"Line number: {line_number}", flush=True)

    async def get_mnist_data(self, iteration: int) -> object:
        """Get MNIST data."""
        print(f"MnistActor.get_mnist_data", flush=True)
        has_value, val = await self._state_manager.try_get_state(
            f"mnist_state_backup_{iteration}"
        )
        return val

    async def run(self) -> object:
        """An actor method which starts the timer."""
        print(f"MnistActor.finish", flush=True)
        has_value, data = await self._state_manager.try_get_state("mnist_timer_state")
        data["start"] = time.time()
        await self._state_manager.set_state("mnist_timer_state", data)
        await self._state_manager.save_state()
        return data

    async def finish(self) -> object:
        """An actor method which stops the timer."""
        print(f"MnistActor.finish", flush=True)
        has_value, data = await self._state_manager.try_get_state("mnist_timer_state")
        data["end"] = time.time()
        await self._state_manager.set_state("mnist_timer_state", data)
        await self._state_manager.save_state()
        return data
