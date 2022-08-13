from dapr.actor import ActorInterface, actormethod


class BankActorInterface(ActorInterface):
    @actormethod(name="SetUp")
    async def set_up(self) -> None:
        ...

    @actormethod(name="Transfer")
    async def transfer(self, data) -> None:
        ...

    @actormethod(name="Run")
    async def run(self) -> object:
        ...

    @actormethod(name="Finish")
    async def finish(self) -> object:
        ...
