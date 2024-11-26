from formatters import LatexFormatter, Rephraser
from notion_api import NotionAPI

if __name__ == "__main__":
    notion_api = NotionAPI()
    blocks = notion_api.read_blocks()
    rephraser = Rephraser(notion_api)
    rephraser.rephrase_blocks(blocks)
