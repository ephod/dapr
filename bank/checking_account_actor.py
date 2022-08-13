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
from uuid import uuid4

from checking_account_actor_interface import CheckingAccountActorInterface
from dapr.actor import Actor, Remindable


class CheckingAccountData(TypedDict):
    balance: int
    checking_account: str
    customer: str
    uuid: str


class CheckingAccountActor(Actor, CheckingAccountActorInterface, Remindable):
    def __init__(self, ctx, actor_id):
        super(CheckingAccountActor, self).__init__(ctx, actor_id)

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

    async def set_up(self, data: CheckingAccountData) -> None:
        """An actor method to set up class."""
        print(f"CheckingAccountActor.set_up", flush=True)
        data["uuid"] = str(uuid4())
        await self._state_manager.set_state("checking_account_actor", data)
        await self._state_manager.save_state()

    async def deposit(self, amount: int) -> None:
        """Deposit request."""
        print(f"CheckingAccountActor.deposit", flush=True)
        val: CheckingAccountData
        has_value, val = await self._state_manager.try_get_state(
            "checking_account_actor"
        )
        val["balance"] += amount
        print(
            f"Receipt for depositing {amount} into checking account {val['uuid']}",
            flush=True,
        )
        await self._state_manager.set_state("checking_account_actor", val)
        await self._state_manager.save_state()

    async def show_balance(self) -> int:
        """Show balance request."""
        print(f"CheckingAccountActor.show_balance", flush=True)
        val: CheckingAccountData
        has_value, val = await self._state_manager.try_get_state(
            "checking_account_actor"
        )
        print(
            f"Current balance {val['balance']} within checking account {val['uuid']}",
            flush=True,
        )
        return val["balance"]

    async def withdrawal(self, amount: int) -> int:
        """Withdrawal request."""
        print(f"CheckingAccountActor.withdrawal", flush=True)
        val: CheckingAccountData
        has_value, val = await self._state_manager.try_get_state(
            "checking_account_actor"
        )
        if val["balance"] >= amount:
            val["balance"] -= amount
            print(
                f"Withdrawal {amount}, your new balance is {val['balance']} within checking account {val['uuid']}"
            )
            await self._state_manager.set_state("checking_account_actor", val)
            await self._state_manager.save_state()
            return -1
        else:
            print(
                f"Not enough funds within checking account {val['uuid']}. "
                f"Amount {amount} is greater than {val['balance']} balance. "
                f"Trying to withdraw from savings account",
                flush=True,
            )
            return amount
