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

from dapr.actor import ActorId, ActorProxy
from judge_actor_interface import JudgeActorInterface
from pinger_actor_interface import PingerActorInterface
from ponger_actor_interface import PongerActorInterface


async def main():
    # Create proxy clients
    judge_proxy = ActorProxy.create("JudgeActor", ActorId("1"), JudgeActorInterface)
    ping_proxy = ActorProxy.create("PingerActor", ActorId("2"), PingerActorInterface)
    pong_proxy = ActorProxy.create("PongerActor", ActorId("3"), PongerActorInterface)

    PING_PONG_GAMES = 1_000
    print("JudgeActor.set_up", flush=True)
    print(f"Game number: {PING_PONG_GAMES}", flush=True)
    await judge_proxy.SetUp(PING_PONG_GAMES)

    print("JudgeActor.ping_ready", flush=True)
    is_ping_ready = await judge_proxy.PingReady()

    print("JudgeActor.pong_ready", flush=True)
    is_pong_ready = await judge_proxy.PongReady()

    if is_ping_ready and is_pong_ready:
        print("JudgeActor.run", flush=True)
        rtn_obj = await judge_proxy.Run()
        print(f"Return object: {rtn_obj}", flush=True)

        for i in range(rtn_obj["num_pings"]):
            await ping_proxy.Pong()
            await pong_proxy.Ping()

        print("JudgeActor.finish", flush=True)
        rtn_obj = await judge_proxy.Finish()

        total = rtn_obj["end"] - rtn_obj["start"]
        print(f"Did {rtn_obj['num_pings']} pings in {total} s", flush=True)
        print(f"{rtn_obj['num_pings'] / total} pings per second", flush=True)

    print("Please stop", flush=True)


asyncio.run(main())
