def convert_latex(blocks):
    LATEX_DELIMITERS = [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")]

    def process_text_content(rich_text):
        content = rich_text["text"]["content"]
        start_idx = 0
        result = []

        while start_idx < len(content):
            found_delim = False
            for start_delim, end_delim in LATEX_DELIMITERS:
                start_pos = content.find(start_delim, start_idx)
                if start_pos != -1:
                    end_pos = content.find(end_delim, start_pos + len(start_delim))
                    if end_pos != -1:
                        # Add text before equation if exists
                        if start_pos > start_idx:
                            before_text = content[start_idx:start_pos]
                            text_part = {"type": "text", "text": {"content": before_text, "link": None}, "plain_text": before_text, "annotations": rich_text["annotations"], "href": rich_text["href"]}
                            result.append(text_part)

                        # Add equation part
                        equation_content = content[start_pos + len(start_delim) : end_pos].strip()
                        equation_part = {"type": "equation", "equation": {"expression": equation_content}, "plain_text": equation_content, "annotations": rich_text["annotations"], "href": rich_text["href"]}
                        result.append(equation_part)

                        start_idx = end_pos + len(end_delim)
                        found_delim = True
                        break

            if not found_delim:
                # Add remaining text
                remaining_text = content[start_idx:]
                text_part = {"type": "text", "text": {"content": remaining_text, "link": None}, "plain_text": remaining_text, "annotations": rich_text["annotations"], "href": rich_text["href"]}
                result.append(text_part)
                break

        return result

    def process_rich_text(rich_text_list):
        result = []
        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                result.extend(process_text_content(rich_text))
            else:
                result.append(rich_text)
        return result

    for block in blocks:
        rich_text = block[block["type"]]["rich_text"]
        if rich_text:
            block[block["type"]]["rich_text"] = process_rich_text(rich_text)
    return blocks


blocks = [
    {
        "object": "block",
        "id": "144bb2c6-d133-804d-abba-f25e60d0e8fc",
        "parent": {"type": "page_id", "page_id": "143bb2c6-d133-8045-9053-f33d84fd6cdb"},
        "created_time": "2024-11-20T08:34:00.000Z",
        "last_edited_time": "2024-11-20T08:34:00.000Z",
        "created_by": {"object": "user", "id": "3c876d9c-6506-491d-ae3f-65fa9933efaa"},
        "last_edited_by": {"object": "user", "id": "3c876d9c-6506-491d-ae3f-65fa9933efaa"},
        "has_children": False,
        "archived": False,
        "in_trash": False,
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"type": "text", "text": {"content": "A function ", "link": None}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"}, "plain_text": "A function ", "href": None},
                {"type": "equation", "equation": {"expression": "f(x)"}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"}, "plain_text": "f(x)", "href": None},
                {"type": "text", "text": {"content": " is convex if, for any two points \\( x_1 \\) and \\( x_2 \\) in its domain and any \\( \\lambda \\) such that \\( 0 \\leq \\lambda \\leq 1 \\), the following holds", "link": None}, "annotations": {"bold": False, "italic": False, "strikethrough": False, "underline": False, "code": False, "color": "default"}, "plain_text": " is convex if, for any two points \\( x_1 \\) and \\( x_2 \\) in its domain and any \\( \\lambda \\) such that \\( 0 \\leq \\lambda \\leq 1 \\), the following holds", "href": None},
            ],
            "color": "default",
        },
    },
]

print(convert_latex(blocks))
