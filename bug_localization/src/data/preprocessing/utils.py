import re


def is_utf_8(text: str) -> int:
    try:
        encoded = text.encode('utf-8')
    except UnicodeEncodeError:
        return False
    else:
        return True


def has_media_in_text(issue_body: str) -> bool:
    try:
        # URL
        images = re.findall(
            r"!\[.*?\]\((.*?\.(jpg|png|gif|jpeg|svg|bmp|tiff|webp|heic|psd|raw|mp3|mp4|mov|wmv|avi|mkv))\)",
            issue_body,
            re.I
        )
        # HTML
        images += re.findall(
            r'src="(.*?\.(jpg|png|gif|jpeg|svg|bmp|tiff|webp|heic|psd|raw|mp3|mp4|mov|wmv|avi|mkv))"',
            issue_body,
            re.I
        )
        return len(images) > 0
    except Exception as e:
        print("Can not parse images from text", e)
        return False


def remove_comments(text: str) -> str:
    # Regex pattern to match comments (starting with `<!--` and ending with `-->`)
    comment_pattern = r"<!--[\s\S]*?-->"
    text = re.sub(comment_pattern, "", text)
    return text


def remove_code(text: str) -> str:
    # Regex pattern to match code blocks (starting with ``` and ending with ```)
    code_pattern = r"```[\s\S]*?```"
    text = re.sub(code_pattern, "", text)
    return text


def get_links(text: str) -> list[str]:
    url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
    urls = re.findall(url_pattern, text)
    return urls


def get_code_blocks(text: str) -> list[str]:
    code_pattern = re.compile(r'```(.*?)```', re.DOTALL)
    code_blocks = re.findall(code_pattern, text)

    return code_blocks
