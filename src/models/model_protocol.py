
from __future__ import annotations

from typing import Protocol, Any, runtime_checkable
from deriva_ml.execution import Execution
from deriva_ml import DerivaML

@runtime_checkable
class DerivaMLModel(Protocol):
    def __call__(self,
                 *args: Any,
                 ml_instance: DerivaML,
                 execution: Execution,
                 **kwargs: Any) -> None:
        """
        Prototype for function that interfaces between DerivaML and underlying ML framework
        """
        ...
