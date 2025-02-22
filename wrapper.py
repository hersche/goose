#!/usr/bin/env python3
"""
This script calls the Gemini API to generate content based on a provided prompt.
It fetches the API key either from the command-line argument '--api-key' (which takes precedence)
or from the environment variable 'GEMINI_API_KEY'. It parses the prompt passed via the
command-line argument '--prompt' and outputs a structured JSON result.

It also enables code execution by configuring the code_execution tool.
Any parts of the response that include text, generated code, execution output,
or inline data (e.g. images) are processed and structured by the helper function
'display_code_execution_result'.

Refer to:
  - Gemini API Thinking: https://ai.google.dev/gemini-api/docs/thinking
  - Gemini API Code Execution: https://ai.google.dev/gemini-api/docs/code-execution?lang=python
"""

import os
import argparse
import json
from google import genai
from google.genai import types



def display_code_execution_result(response):
    """
    Extracts and structures output from a Gemini API response with code execution enabled.

    Parameters:
      response: A Gemini API response object.

    Returns:
      A list of dictionaries. Each dictionary may include:
        - 'text': plain text output (if available)
        - 'code': code string from executable_code (if available)
        - 'language': language of the executable code (if available)
        - 'execution_result': output from the executed code (if available)
        - 'inline_data': dict with 'data' and 'mime_type' for any inline data (if available)
    """
    results = []
    candidate = response.candidates[0]
    global responseText
    responseText = ""
    for part in candidate.content.parts:
        part_result = {}
        if part.text is not None:
            part_result['text'] = part.text
            responseText += part.text
        if getattr(part, 'executable_code', None) is not None:
            part_result['code'] = part.executable_code.code
            responseText += part.executable_code.code
            if hasattr(part.executable_code, 'language'):
                part_result['language'] = part.executable_code.language
        if getattr(part, 'code_execution_result', None) is not None:
            part_result['execution_result'] = part.code_execution_result.output
            responseText += part.code_execution_result.output
        if getattr(part, 'inline_data', None) is not None:
            part_result['inline_data'] = {
                'data': part.inline_data.data,
                'mime_type': getattr(part.inline_data, 'mime_type', None)
            }
        results.append(part_result)
    return results

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description='Generate content using the Gemini API with code execution support.'
    )
    parser.add_argument('--prompt', required=True, help='The prompt to send to the Gemini API')
    parser.add_argument('--api-key', help='API key for Gemini API (beware of bash history leaks)')
    args = parser.parse_args()

    # Prefer the API key from the command-line argument over the environment variable.
    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("API key not provided. Use --api-key or set GEMINI_API_KEY environment variable.")

    # Initialize the Gemini client using the API key and HTTP options.
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    # Configure code execution tool via the SDK.
    config = types.GenerateContentConfig(
        tools=[types.Tool(code_execution=types.ToolCodeExecution())]
    )

    # Generate content with the provided prompt and code execution enabled.
    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=args.prompt,
        config=config
    )

    # Extract structured code execution output (if any).
    structured_parts = display_code_execution_result(response)

    # Prepare structured output.
    output = {
        "prompt": args.prompt,
        "structured_response": structured_parts,
        "model": "gemini-2.0-flash-thinking-exp",
        "status": "success"
    }

    # Output the structured result as JSON.
    print(responseText)

if __name__ == '__main__':
    main()
