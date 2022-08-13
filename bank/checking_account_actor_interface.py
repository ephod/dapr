from dapr.actor import ActorInterface, actormethod


class CheckingAccountActorInterface(ActorInterface):
    @actormethod(name="SetUp")
    async def set_up(self, data) -> None:
        ...

    @actormethod(name="Deposit")
    async def deposit(self, amount: int) -> None:
        ...

    @actormethod(name="ShowBalance")
    async def show_balance(self) -> int:
        ...

    @actormethod(name="Withdrawal")
    async def withdrawal(self, amount: int) -> int:
        ...
