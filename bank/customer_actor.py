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
from typing import Optional, TypedDict

from customer_actor_interface import CustomerActorInterface
from dapr.actor import Actor, Remindable


class CustomerData(TypedDict):
    checking_account: str
    savings_account: str


class CustomerActor(Actor, CustomerActorInterface, Remindable):
    def __init__(self, ctx, actor_id):
        super(CustomerActor, self).__init__(ctx, actor_id)

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

    async def set_up(self, data: CustomerData) -> None:
        """An actor method to set up class."""
        print(f"CustomerActor.set_up", flush=True)
        await self._state_manager.set_state("customer_actor", data)
        await self._state_manager.save_state()

    async def withdrawal(self, amount: int) -> int:
        """Withdrawal."""
        print(f"CustomerActor.withdrawal", flush=True)
        has_value, val = await self._state_manager.try_get_state("customer_actor")
        checking_account = val["checking_account"]
        print(
            f"Customer withdraws {amount} from {checking_account} checking account",
            flush=True,
        )
        return amount

    async def deposit(self, amount: int) -> int:
        """Deposit."""
        print(f"CustomerActor.deposit", flush=True)
        has_value, val = await self._state_manager.try_get_state("customer_actor")
        checking_account = val["checking_account"]
        print(
            f"Customer deposits {amount} from {checking_account} checking account",
            flush=True,
        )
        return amount
