from abc import ABC, abstractmethod


class DiscreteModule(ABC):
    CAN_USE_DISCRETE_EVAL = True

    @abstractmethod
    def switch_mode(self, *, discrete: bool): ...
