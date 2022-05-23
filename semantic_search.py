from typing import Optional
import torch
from shared_objects import cross_encoder, question_answer_model, question_answer_tokenizer
from models import SemanticSearchResponse, \
    SemanticSearchRequest, \
    CrossEncodeInput, \
    CrossEncodeOutput, \
    Document, \
    AnswerQuestionResponse


def answer_question(question: str, context: Document, score: float) -> Optional[AnswerQuestionResponse]:
    inputs = question_answer_tokenizer(
        question,
        context.text,
        add_special_tokens=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = question_answer_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    #
    # Get the most likely beginning of answer with the argmax of the score
    #
    answer_start = torch.argmax(answer_start_scores)

    #
    # Get the most likely end of answer with the argmax of the score
    #
    answer_end = torch.argmax(answer_end_scores) + 1

    answer_as_tokens = question_answer_tokenizer.convert_ids_to_tokens(
        input_ids[answer_start:answer_end]
    )
    if answer_end <= answer_start or answer_end is None or answer_start is None:
        return None

    #
    # create a tag that wraps around the answer as highlight
    #
    input_as_tokens = question_answer_tokenizer.convert_ids_to_tokens(input_ids)
    answer_text = question_answer_tokenizer.convert_tokens_to_string(answer_as_tokens)
    answer_length = answer_end - answer_start
    input_as_tokens.insert(answer_start, "[START]")
    input_as_tokens.insert(answer_end + answer_length + 1, "[END]")
    answer_highlighted = question_answer_tokenizer.convert_tokens_to_string(input_as_tokens)

    return AnswerQuestionResponse(
        answer_text=answer_text,
        answer_highlighted=answer_highlighted,
        document=context,
        score=score
    )


def cross_encode(cross_encode_input: CrossEncodeInput) -> list[CrossEncodeOutput]:
    reference = cross_encode_input.reference
    comparisons: list[Document] = cross_encode_input.comparisons
    model_input = [[reference, doc.text] for doc in comparisons]
    cross_scores = cross_encoder.predict(model_input)
    model_output = list(zip(cross_scores.tolist(), comparisons))
    ret = [
        CrossEncodeOutput(score=output[0], document=output[1]) for output in
        sorted(model_output, key=lambda x: x[0], reverse=True)
    ]
    return ret


def semantic_search(request: SemanticSearchRequest) -> SemanticSearchResponse:
    question = request.question
    cross_encode_outputs = cross_encode(
        cross_encode_input=CrossEncodeInput(
            reference=question,
            comparisons=request.documents
        )
    )

    #
    # choose only the top 5 results in the re-ranked output
    #
    top_5_cross_encode_output = cross_encode_outputs[:5]

    all_answers = [answer_question(question=question, context=candidate.document, score=candidate.score) for candidate
                   in
                   top_5_cross_encode_output
                   ]
    answer_candidates = [answer for answer in all_answers if answer is not None]
    return SemanticSearchResponse(
        answer_candidates=answer_candidates
    )
