read_fs_tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads file by given path. "
                           "Returns text of the file or None if file does not exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path from content root to the file to be read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Gets a list of files and directories within given directory path. "
                           "Returns list of files and directories names, or None if given directory does not exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path from content root to the directory to be listed without leading /. "
                                       "In case of root directory use empty string"
                    }
                },
                "required": ["path"]
            }
        }
    }
]
