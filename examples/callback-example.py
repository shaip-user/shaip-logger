# Import the necessary libraries
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Import the LangchainCallbackHandler from the shaip logging library
from shaip_logger.tracing.callback.langchain import LangchainCallbackHandler
from shaip_logger.api_key import ShaipApiKey
import os

# Set the Shaip API Key for logging the traces to Shaip
ShaipApiKey.set_api_key(os.environ['SHAIP_API_KEY'])

# Create a prompt template for the chat
system_template = (
    '''You are a helpful assistant who generates lines about a particular topic'''
)
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
template = '''Write a line on the following topic: {text} Your response:'''
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, HumanMessagePromptTemplate.from_template(template)]
)

# Create a chain and add the LangchainCallbackHandler as a callback
chain1 = LLMChain(
    llm=ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY']),
    prompt=chat_prompt,
)
response = chain1.invoke('AI', {"callbacks": [LangchainCallbackHandler()]})
print("Response:", response)

# The response will be printed in the console and the trace will be available in the Shaip UI
