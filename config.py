from pydantic import BaseModel


class ItemSelection(BaseModel):
    src: str
    name: str


class SelectionConfig(BaseModel):
    items: list[ItemSelection]
