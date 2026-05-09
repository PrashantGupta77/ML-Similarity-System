from pydantic import BaseModel
from typing import List


class StoreInput(BaseModel):
    text: str


class SearchInput(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]