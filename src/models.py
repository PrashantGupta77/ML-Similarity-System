from pydantic import BaseModel, Field
from typing import List


class StoreInput(BaseModel):
    text: str


class SearchInput(BaseModel):

    query: str

    top_k: int = Field(
        default=5,
        gt=0,
        le=20
    )


class SearchResult(BaseModel):
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]