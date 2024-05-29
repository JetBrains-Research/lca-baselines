AGENT_PROMPT_TEMPLATE = """
    Issue: {}
    You are given bug issue description.
    Select subset of 1-5 files which SHOULD be fixed according to issue.
    Start with repo exploration by listing files in directory (start with root). 
    Read bug-related files to make sure they contain bugs.
    Provide output in JSON format with one field 'files' with list of file names which SHOULD be fixed.
    Provide ONLY json without any additional comments.
"""
