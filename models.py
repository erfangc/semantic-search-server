import pydantic
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    text: str
    metadata: dict


class SingleSemanticSimilarityResponse(BaseModel):
    document: Document
    score: float


class SemanticSimilarityResponse(BaseModel):
    results: list[SingleSemanticSimilarityResponse]


class AnswerQuestionResponse(BaseModel):
    answer_text: str
    answer_highlighted: str
    score: float
    document: Document


class SemanticSearchResponse(BaseModel):
    answer_candidates: list[AnswerQuestionResponse]


class SemanticSearchRequest(BaseModel):
    question: str
    documents: list[Document]


class CrossEncodeOutput(BaseModel):
    document: Document
    score: float


class CrossEncodeInput(BaseModel):
    reference: str
    comparisons: list[Document]


class SemanticSimilarityRequest(BaseModel):
    documents: list[Document]
    query: str
