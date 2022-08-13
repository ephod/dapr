from dapr.actor import ActorInterface, actormethod


class MnistActorInterface(ActorInterface):
    @actormethod(name="SetUp")
    async def set_up(self, learning_rate: float) -> None:
        ...

    @actormethod(name="ForwardBackwardPass")
    async def forward_backward_pass(self, data) -> None:
        ...

    @actormethod(name="Validate")
    async def validate(self, data) -> None:
        ...

    @actormethod(name="GetMnistData")
    async def get_mnist_data(self, iteration: int) -> object:
        ...

    @actormethod(name="BackupMnistData")
    async def backup_mnist_data(self, iteration: int) -> None:
        ...

    @actormethod(name="Run")
    async def run(self) -> object:
        ...

    @actormethod(name="Finish")
    async def finish(self) -> object:
        ...
