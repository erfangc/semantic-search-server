import spacy
from sentence_transformers import SentenceTransformer, util

from models import SemanticSimilarityResponse, SemanticSimilarityRequest, SingleSemanticSimilarityResponse, Document

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sentence_score(text: str, query: str) -> float:
    doc = nlp(text)

    comparisons = [sentence.__str__() for sentence in doc.sents]
    query = [query]

    comparison_embeddings = model.encode(comparisons, convert_to_tensor=True)
    query_embeddings = model.encode(query, convert_to_tensor=True)

    # -----------------
    # cosine similarity
    # -----------------
    cosine_scores = util.cos_sim(
        comparison_embeddings,
        query_embeddings,
    )

    for i in range(len(comparisons)):
        for j in range(len(query)):
            print("{} Score: {:.4f}".format(comparisons[i], cosine_scores[i][j]))

    # -----------------------------------------------
    # filter out any sentences < 0.4 and sum the rest
    # -----------------------------------------------
    score = 0
    for i in range(len(comparisons)):
        if cosine_scores[i][0] > 0.4:
            score = score + cosine_scores[i][0]

    print("Total score: {:.4f}".format(score))
    return score


def semantic_similarity(request: SemanticSimilarityRequest) -> SemanticSimilarityResponse:
    query = request.query
    results = [
        single_semantic_similarity_response(score=get_sentence_score(doc.text, query), document=doc) for doc in
        request.documents]
    ret = SemanticSimilarityResponse()
    ret.results = results
    return ret


def single_semantic_similarity_response(score: float, document: Document) -> SingleSemanticSimilarityResponse:
    ret = SingleSemanticSimilarityResponse()
    ret.score = score
    ret.document = document
    return ret
