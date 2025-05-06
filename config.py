from pathlib import Path

from pydantic import BaseModel


class Action(BaseModel):
    name: str
    preconditions: list[str]
    add_list: list[str]
    delete_list: list[str]


class Item(BaseModel):
    id: int
    path: Path
    name: str


class StripsConfig(BaseModel):
    items: list[Item]
    actions: list[Action]
    time_steps: int
    inventory_size: int
