import json
import re


def extract_json_from_output(llm_output):
    """
    Tries to extract a JSON object from LLM output.
    Handles plain JSON, JSON with code blocks, and noisy outputs.

    Args:
        llm_output (str): The LLM-generated output containing JSON.

    Returns:
        dict or list: Parsed JSON as Python object (dict or list),
                      or None if no valid JSON is found.
    """

    # Regex for JSON object and array patterns
    json_pattern = r"({.*?}|\[.*?\])"

    # Regex for Markdown-style code blocks
    code_block_pattern = r"```(?:json)?\s*(.*?)```"

    # Extract JSON from code blocks
    code_blocks = re.findall(code_block_pattern, llm_output, flags=re.DOTALL)

    # Add the entire output as a fallback candidate
    candidates = code_blocks if code_blocks else [llm_output]

    # Search for JSON patterns in each candidate
    for candidate in candidates:
        json_matches = re.findall(json_pattern, candidate, flags=re.DOTALL)
        for json_match in json_matches:
            try:
                # Attempt to parse each JSON match
                parsed_json = json.loads(json_match.strip())
                return parsed_json  # Return the first valid JSON object
            except json.JSONDecodeError:
                continue

    # If no valid JSON is found, return None
    return None


# Example Usage
if __name__ == "__main__":
    llm_outputs = [
        # Plain JSON
        '{"key": "value", "number": 123}',

        # JSON embedded in a code block
        "Here is the JSON: ```json\n{\"key\": \"value\", \"number\": 123}\n```",

        # Noisy output with inline JSON
        "Some irrelevant text... {\"key\": \"value\", \"number\": 123} ...more text here...",

        # Invalid JSON
        "```json\n{invalid: json here}```",

        # Multiple JSON patterns
        "First JSON: {\"key\": \"value1\"}, second JSON: {\"key\": \"value2\"}",

        # JSON array
        "Here is a JSON array: ```json\n[1, 2, 3, 4]\n```"
    ]

    for output in llm_outputs:
        print("LLM Output:")
        print(output)
        print("\nParsed JSON:")
        result = extract_json_from_output(output)
        print(result)
        print("-" * 50)
