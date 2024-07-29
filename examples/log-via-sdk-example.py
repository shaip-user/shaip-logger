import os

from openai import OpenAI

from shaip_logger.api_key import ShaipApiKey
from shaip_logger.inference_logger import InferenceLogger
from shaip_logger.exception.custom_exception import CustomException

ShaipApiKey.set_api_key(os.getenv('SHAIP_API_KEY'))

messages = [{"role": "user", "content": "What is machine learning?"}]

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=messages,
)

try:
    InferenceLogger.log_inference(
        prompt_slug="sdk_test",
        prompt=messages,
        language_model_id="gpt-4-1106-preview",
        response=response.model_dump(),
        external_reference_id="abc",
        cost=0.0123,
        custom_attributes={
            "name": "John Doe"
            # Your custom attributes
        },
    )
except Exception as e:  # pylint: disable=broad-exception-caught
    if isinstance(e, CustomException):
        print(e.status_code)
        print(e.message)
    else:
        print(e)
