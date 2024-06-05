import re


def camel_case_split(identifier):
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def snake_case_split(identifier):
    return identifier.split("_")


def split_identifier(identifier):
    parts = [p.lower() for part in snake_case_split(identifier) for p in camel_case_split(part) if p != ""]
    return parts
