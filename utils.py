def read_html_file(file_path):
    try:
        with open(file_path, "r") as file:
            html_content = file.read()
        return html_content
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None
