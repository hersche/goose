#!/usr/bin/env python3
"""
This script calls the Gemini API to generate content based on a provided prompt.
It fetches the API key from '--api-key' or the 'GEMINI_API_KEY' environment variable.
It outputs a JSON object in the OpenAI API-compatible format expected by the provider.
"""

import os
import argparse
import json
import time
import sys
from google import genai
from google.genai import types

def display_code_execution_result(response):
    candidate = response.candidates[0]
    global responseText
    responseText = ""
    for part in candidate.content.parts:
        if part.text is not None:
            responseText += part.text
        if getattr(part, 'executable_code', None) is not None:
            responseText += part.executable_code.code
        if getattr(part, 'code_execution_result', None) is not None:
            responseText += part.code_execution_result.output

def main():
    parser = argparse.ArgumentParser(
        description='Generate content using the Gemini API with code execution support.'
    )
    parser.add_argument('--prompt', required=True, help='The prompt to send to the Gemini API')
    parser.add_argument('--api-key', help='API key for Gemini API')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("API key not provided. Use --api-key or set GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=args.prompt,
        config=config
    )

    display_code_execution_result(response)

    prompt_tokens = len(args.prompt.split())
    completion_tokens = len(responseText.split())
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

    result = {
        "id": "gemini-2.0-flash-thinking-exp",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "gemini-2.0-flash-thinking-exp",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": responseText
                },
                "finish_reason": "stop"
            }
        ],
        "usage": usage
    }

    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
