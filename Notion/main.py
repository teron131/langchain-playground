from formatters import LatexFormatter
from notion_api import NotionAPI

if __name__ == "__main__":
    notion_api = NotionAPI()
    content = notion_api.read_blocks_markdown()
    print(content)
