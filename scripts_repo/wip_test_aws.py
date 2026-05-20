#!/usr/bin/env python
from openai import OpenAI
import boto3
import json
import sys
import os

from openai import OpenAI

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", api_key=os.getenv("NVIDIA_NIM_KEY")
)


completion = client.chat.completions.create(
    model="qwen/qwen3-next-80b-a3b-thinking",
    messages=[
        {"role": "user", "content": "Explain how quicksort works on a small example"}
    ],
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
    reasoning_effort="medium",
    stream=False,
)

reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
print("-" * 80)
if reasoning:
    print(reasoning)
print("-" * 80)
print(completion.choices[0].message.content)

# response = client.responses.create(
#     model="qwen/qwen3-next-80b-a3b-thinking",
#     reasoning={"effort": "medium"},
#     input=[
#         {"role": "user", "content": "Explain how quicksort works on a small example"}
#     ],
# )

# print(response)


sys.exit(0)

# This uses your AWS credentials from your environment, so ensure those are set up correctly. For some
# reason my bedrock token wasn't working.
client = boto3.client("bedrock-runtime", region_name="us-east-1")
# # aws_access_key_id='',
# # aws_secret_access_key='')

# Configure OpenAI client for Bedrock
# openapiClient = OpenAI(
#     base_url="https://bedrock-mantle.{region}.amazonaws.com",  # Replace {region} with your AWS region
#     api_key="your-aws-access-key-id",
#     # Note: You'll also need to handle AWS authentication properly
# )

# # Example for us-east-1
# openapiClient = OpenAI(
#     base_url="https://bedrock-mantle.us-east-1.amazonaws.com",
#     api_key="your-aws-access-key-id"
# )


# response = openapiClient.responses.create(
#     model="qwen.qwen3-next-80b-a3b",
#     input=[
#         {"role": "user", "content": "Write a one-sentence bedtime story about a unicorn."}
#     ]
# )

####################################################################################################

# system_prompt = """You always reply with a single JSON object. No markdown fences, no commentary
# outside the JSON. If the user asks for fields, include exactly those fields and no others.
# """
# user_prompt = """Extract the author and year from this sentence: "Pride and Prejudice was written by
# Jane Austen and published in 1813." Return JSON with keys "author" and "year"."""

# response = client.invoke_model(
#     modelId="qwen.qwen3-next-80b-a3b",
#     body=json.dumps(
#         {
#             "messages": [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             "max_tokens": 100,
#             # "response_format": {
#             #     "type": "json_schema",
#             #     "json_schema": {
#             #         "name": "data_extraction",
#             #         "schema": {
#             #             "type": "object",
#             #             "properties": {
#             #                 "Author": {"type": "string", "description": "Author"},
#             #                 "Year": {"type": "string", "description": "Year"}
#             #             },
#             #             "required": ["Author", "Year"],
#             #             "additionalProperties": False
#             #         }
#             #     }
#             # }
#         }
#     ),
# )


# print(response["body"].read().decode("utf-8"))

####################################################################################################
# This was to test odd behaviour I was seeing in bedrock. When adding just this extra system prompt
# it caused the json output type of the year to change from an int to a string.

# system_prompt_extra = """\n\nYou have access to tools. Call a tool when it would help answer the
# user. Do not fabricate tool results."""

# response = client.invoke_model(
#     modelId="qwen.qwen3-next-80b-a3b",
#     body=json.dumps(
#         {
#             "messages": [
#                 {"role": "system", "content": system_prompt + system_prompt_extra},
#                 {"role": "user", "content": user_prompt},
#             ],
#             "max_tokens": 100,
#         }
#     ),
# )

# print(response["body"].read().decode("utf-8"))

####################################################################################################

# response = client.invoke_model(
#     modelId='qwen.qwen3-next-80b-a3b',
#     body=json.dumps({
#         'messages': [
#             {'role': 'user', 'content': 'Instructions for a good cup of coffee'}
#         ],
#         'max_tokens': 2048,        # Customize output length
#         'temperature': 1.0,        # Control creativity
#         'top_p': 0.9,             # Control diversity
#         'stop_sequences': ['\n\n', 'END']  # Stop generation at these sequences
#     })
# )

# result = json.loads(response['body'].read())
# print(result)

####################################################################################################
# CONVERSATIONS
####################################################################################################

####################################################################################################
# Conversations with structured output
####################################################################################################

# # Define JSON schema
# schema = {
#     "type": "object",
#     "properties": {
#         "title": {"type": "string", "description": "The title"},
#         "summary": {"type": "string", "description": "Brief summary"},
#     },
#     "required": ["title", "summary"],
#     "additionalProperties": False,
# }

# response = client.converse(
#     modelId="qwen.qwen3-next-80b-a3b",
#     messages=[{"role": "user", "content": [{"text": "Hello"}]}],
#     inferenceConfig={"maxTokens": 100, "temperature": 1.0},
#     outputConfig={
#         "textFormat": {
#             "type": "json_schema",
#             "structure": {
#                 "jsonSchema": {
#                     "schema": json.dumps(schema),  # Must be a JSON string
#                     "name": "greeting_response",
#                     "description": "Response schema for greetings",
#                 }
#             },
#         }
#     },
# )

# # Extract the text from the response
# text = response["output"]["message"]["content"][0]["text"]
# result = json.loads(text)
# print(result)


####################################################################################################
# Conversations with reasoning
####################################################################################################

# response = client.converse(
#     modelId="qwen.qwen3-next-80b-a3b",
#     messages=[
#         {
#             "role": "user",
#             "content": [{"text": "Explain how quicksort works on a small example."}],
#         }
#     ],
#     inferenceConfig={
#         "maxTokens": 512,
#         "temperature": 0.3,
#     },
# )

# text = response["output"]["message"]["content"][0]["text"]
# print("No reasoning:")
# print(text)

# print(response["output"])

response = client.converse(
    modelId="qwen.qwen3-next-80b-a3b",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Explain how quicksort works on a small example."}],
        }
    ],
    # additionalModelRequestFields={
    #     # Qwen-specific reasoning flags via provider extension
    #     "thinking": True,  # enable reason/think mode [web:20]
    #     "thinkBudget": 1024,  # max tokens for reasoning content [web:20]
    # },
    additionalModelRequestFields={
        "reasoningConfig": {
            "type": "enabled",  # or "disabled" [web:30]
            "budget_tokens": 2048,  # required when enabled [web:30]
        }
    },
    inferenceConfig={
        "maxTokens": 512,
        "temperature": 0.3,
    },
)

text = response["output"]["message"]["content"][0]["text"]
print("With reasoning:")
print(text)

print(response["output"])
