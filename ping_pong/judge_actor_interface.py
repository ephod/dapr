from dapr.actor import ActorInterface, actormethod


class JudgeActorInterface(ActorInterface):
    @actormethod(name="SetUp")
    async def set_up(self, num_pings: int) -> None:
        ...

    @actormethod(name="PingReady")
    async def ping_ready(self) -> bool:
        ...

    @actormethod(name="PongReady")
    async def pong_ready(self) -> bool:
        ...

    @actormethod(name="Run")
    async def run(self) -> object:
        ...

    @actormethod(name="Finish")
    async def finish(self) -> object:
        ...
