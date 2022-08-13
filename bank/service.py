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

from bank_actor import BankActor
from checking_account_actor import CheckingAccountActor
from customer_actor import CustomerActor
from dapr.actor.runtime.config import (
    ActorReentrancyConfig,
    ActorRuntimeConfig,
    ActorTypeConfig,
)
from dapr.actor.runtime.runtime import ActorRuntime
from dapr.ext.fastapi import DaprActor  # type: ignore
from fastapi import FastAPI  # type: ignore
from savings_account_actor import SavingsAccountActor

app = FastAPI(title=f"{BankActor.__name__}Service")

# This is an optional advanced configuration which enables reentrancy only for the
# specified actor type. By default, reentrancy is not enabled for all actor types.
config = ActorRuntimeConfig()  # init with default values
config.update_actor_type_configs(
    [
        ActorTypeConfig(
            actor_type=BankActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
        ActorTypeConfig(
            actor_type=CustomerActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
        ActorTypeConfig(
            actor_type=CheckingAccountActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
        ActorTypeConfig(
            actor_type=SavingsAccountActor.__name__,
            reentrancy=ActorReentrancyConfig(enabled=True),
        ),
    ]
)
ActorRuntime.set_actor_config(config)

# Add Dapr Actor Extension
actor = DaprActor(app)


@app.on_event("startup")
async def startup_event():
    # Register virtual actors
    await actor.register_actor(BankActor)
    await actor.register_actor(CustomerActor)
    await actor.register_actor(CheckingAccountActor)
    await actor.register_actor(SavingsAccountActor)
