from dapr.actor import ActorInterface, actormethod


class CustomerActorInterface(ActorInterface):
    @actormethod(name="SetUp")
    async def set_up(self, num_pings: int) -> None:
        ...

    @actormethod(name="Deposit")
    async def deposit(self, amount: int) -> int:
        ...

    @actormethod(name="Withdrawal")
    async def withdrawal(self, amount: int) -> int:
        ...
