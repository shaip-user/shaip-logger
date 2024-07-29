from typing import Annotated, Any, Optional
from pydantic import BaseModel, Field
import os
import requests


class Prompt(BaseModel):
    role: str
    content: str


class Context(BaseModel):
    information: list[str]


class Response(BaseModel):
    text: str


class ShaipLogRequest(BaseModel):
    language_model_id: str
    prompt: list[Prompt]
    response: str
    user_query: str
    context: Context
    custom_attributes: dict[str, str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    response_time: int
    prompt_slug: str
    environment: str
    customer_id: str
    customer_user_id: str
    session_id: str
    external_reference_id: Annotated[Optional[str], Field(default=None)]
    expected_response: Annotated[Optional[str], Field(default=None)]
    tools: dict[str, str]
    tool_calls: dict[str, str]
    functions: list[Any]
    function_call_response: dict[str, Any]


def create_request():
    return ShaipLogRequest(
        language_model_id='gpt-4',
        prompt=[
            Prompt(role='system', content='You are math assistant'),
            Prompt(role='user', content='You need to find gcd of 12 and 16'),
        ],
        response='Factors of 12 = 2 x 2 x 3 and 16 = 2 x 2 x 2 x 2.\nHence gcd = 2 x 2 = 4',
        user_query='You need to find gcd of 12 and 16',
        context=Context(information=['Doc1', 'Doc2']),
        custom_attributes={},
        prompt_tokens=23,
        completion_tokens=16,
        total_tokens=39,
        cost=0.002,
        response_time=23,
        prompt_slug='example-api',
        environment='development',
        customer_id='testing-client',
        customer_user_id='123223',
        session_id='20240729-11002002',
        external_reference_id='1122',
        expected_response='',
        tools={},
        tool_calls={},
        functions=[],
        function_call_response={},
    ).model_dump()


API_BASE_URL = 'https://eval.shaip.ai:9000/'
LOG_INFERENCE_URL = f'{API_BASE_URL}/api/v1/log/inference'

info = create_request()
headers = {'x-api-key': os.environ['SHAIP_API_KEY'], 'Content-Type': 'application/json'}
response = requests.post(LOG_INFERENCE_URL, json=info, timeout=60, headers=headers)
print(response)
