from dapr.actor import ActorInterface, actormethod


class PingerActorInterface(ActorInterface):
    @actormethod(name="Pong")
    async def pong(self) -> None:
        ...
