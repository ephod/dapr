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

import datetime
import time
from typing import Optional, TypedDict

from bank_actor_interface import BankActorInterface
from dapr.actor import Actor, Remindable


class TimerData(TypedDict):
    ts: datetime.datetime
    start: float
    end: float


class TransferData(TypedDict):
    amount: int
    from_customer: str
    to_customer: str


class BankActor(Actor, BankActorInterface, Remindable):
    def __init__(self, ctx, actor_id):
        super(BankActor, self).__init__(ctx, actor_id)

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

    async def set_up(self) -> None:
        """An actor method to set up class."""
        print(f"BankActor.set_up", flush=True)
        data: TransferData = {
            "amount": -1,
            "from_customer": "",
            "to_customer": "",
        }
        await self._state_manager.set_state("bank_actor", data)
        await self._state_manager.save_state()
        data: TimerData = {
            "ts": datetime.datetime.now(datetime.timezone.utc),
            "start": -1,
            "end": -1,
        }
        await self._state_manager.set_state("bank_actor_timer", data)
        await self._state_manager.save_state()

    async def transfer(self, data: TransferData) -> None:
        """Transfer."""
        print(f"BankActor.transfer", flush=True)
        from_customer = data["from_customer"]
        to_customer = data["to_customer"]
        print(f"🏦 transfer money from {from_customer} to {to_customer}", flush=True)
        await self._state_manager.set_state("bank_actor", data)
        await self._state_manager.save_state()

    async def run(self) -> object:
        """An actor method which starts the timer."""
        print(f"BankActor.run", flush=True)
        has_value, data = await self._state_manager.try_get_state("bank_actor_timer")
        print(f"Has value: {has_value}", flush=True)
        data["start"] = time.time()
        await self._state_manager.set_state("bank_actor_timer", data)
        await self._state_manager.save_state()
        return data

    async def finish(self) -> object:
        """An actor method which stops the timer."""
        print(f"BankActor.finish", flush=True)
        has_value, data = await self._state_manager.try_get_state("bank_actor_timer")
        print(f"Has value: {has_value}", flush=True)
        data["end"] = time.time()
        await self._state_manager.set_state("bank_actor_timer", data)
        await self._state_manager.save_state()
        return data
