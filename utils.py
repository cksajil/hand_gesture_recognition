def read_markdown_file(markdown_path):
    with open(markdown_path, "r") as file:
        markdown_content = file.read()
    return markdown_content
