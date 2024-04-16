FILE_LIST_PROMPT_TEMPLATE = """
    List of files: {}
    Issue: {}
    You are given a list of files in project and bug issue description.
    Select subset of 1-20 files which SHOULD be fixed according to issue.
    Provide output in JSON format with one field 'files' with list of file names which SHOULD be fixed.
    Provide ONLY json without any additional comments.
"""
