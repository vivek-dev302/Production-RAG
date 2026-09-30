import pytest
from pydantic import ValidationError

from app.schemas import QueryRequest


def test_query_request_accepts_a_normal_question():
    request = QueryRequest(question="What is a primary key?")

    assert request.question == "What is a primary key?"


def test_query_request_rejects_an_empty_question():
    with pytest.raises(ValidationError):
        QueryRequest(question="")
