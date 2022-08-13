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
from typing import Optional

from dapr.actor import Actor, Remindable
from judge_actor_interface import JudgeActorInterface


class JudgeActor(Actor, JudgeActorInterface, Remindable):
    def __init__(self, ctx, actor_id):
        super(JudgeActor, self).__init__(ctx, actor_id)

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

    async def set_up(self, num_pings: int) -> None:
        """An actor method to set up class."""
        print(f"JudgeActor.set_up", flush=True)
        data = {
            "ping_ok": False,
            "pong_ok": False,
            "ts": datetime.datetime.now(datetime.timezone.utc),
            "num_pings": num_pings,
            "start": -1,
            "end": -1,
        }
        await self._state_manager.set_state("judge_state", data)
        await self._state_manager.save_state()

    async def ping_ready(self) -> bool:
        """An actor method which starts ping."""
        print(f"JudgeActor.ping_ready", flush=True)
        has_value, data = await self._state_manager.try_get_state("judge_state")
        data["ping_ok"] = True
        await self._state_manager.set_state("judge_state", data)
        await self._state_manager.save_state()
        return data["ping_ok"]

    async def pong_ready(self) -> bool:
        """An actor method which starts pong."""
        print(f"JudgeActor.pong_ready", flush=True)
        has_value, data = await self._state_manager.try_get_state("judge_state")
        data["pong_ok"] = True
        await self._state_manager.set_state("judge_state", data)
        await self._state_manager.save_state()
        return data["ping_ok"]

    async def run(self) -> object:
        """An actor method which starts the timer."""
        print(f"JudgeActor.run", flush=True)
        has_value, data = await self._state_manager.try_get_state("judge_state")
        data["start"] = time.time()
        await self._state_manager.set_state("judge_state", data)
        await self._state_manager.save_state()
        return data

    async def finish(self) -> object:
        """An actor method which stops the timer."""
        print(f"JudgeActor.finish", flush=True)
        has_value, data = await self._state_manager.try_get_state("judge_state")
        data["end"] = time.time()
        await self._state_manager.set_state("judge_state", data)
        await self._state_manager.save_state()
        return data
