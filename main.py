from fastapi import FastAPI

from models import SemanticSearchResponse, SemanticSearchRequest, SemanticSimilarityRequest, SemanticSimilarityResponse
from semantic_search import semantic_search

app = FastAPI()


@app.post(
    path="/semantic-search-server/api/v1/semantic-search",
    operation_id="semantic_search",
    tags=["semantic-search"],
    response_model=SemanticSearchResponse,
)
def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    return semantic_search(request)


@app.post(
    path="/semantic-search-server/api/v1/semantic-similarity",
    operation_id="semantic_similarity",
    tags=["semantic-search"],
    response_model=SemanticSimilarityResponse,
)
def semantic_similarity(request: SemanticSimilarityRequest) -> SemanticSimilarityResponse:
    return semantic_similarity(request)
