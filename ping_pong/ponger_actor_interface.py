from dapr.actor import ActorInterface, actormethod


class PongerActorInterface(ActorInterface):
    @actormethod(name="Ping")
    async def ping(self) -> None:
        ...
