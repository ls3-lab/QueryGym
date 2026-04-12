from __future__ import annotations
from typing import Dict, Type
from .base import BaseReformulator

METHODS: Dict[str, Type[BaseReformulator]] = {}


def register_method(name: str):
    def deco(cls: Type[BaseReformulator]):
        METHODS[name] = cls
        return cls

    return deco
