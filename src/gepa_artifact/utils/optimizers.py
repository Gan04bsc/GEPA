import dspy

from dataclasses import dataclass
from typing import Type

@dataclass
class OptimizerConfig:
    optimizer: Type[dspy.teleprompt.Teleprompter]
    init_args: dict
    compile_args: dict
    langProBe_configs: dict
    name: str

    def __str__(self):
        return f"""
[[
    Optimizer: {self.name} ({self.optimizer})
    init_args: {self.init_args}
    compile_args: {self.compile_args}
    langProBe_configs: {self.langProBe_configs}
]]
        """

    def __repr__(self):
        return self.__str__()