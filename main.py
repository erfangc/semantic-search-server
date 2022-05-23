from fastapi import FastAPI

from models import SemanticSearchResponse, SemanticSearchRequest
from semantic_search import semantic_search

app = FastAPI()


@app.post(
    path="/semantic-search-server/api/v1/semantic-search",
    operation_id="semantic_search",
    tags=["semantic-search"],
    response_model=SemanticSearchResponse
)
def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    return semantic_search(request)
