from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

fast_llm = ChatOpenAI(model="gpt-4o-mini")
long_context_llm = ChatOpenAI(model="gpt-4o-mini")


direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific."),
        ("user", "{topic}"),
    ]
)


class Subsection(BaseModel):
    subsection_title: str = Field(..., title="Title of the subsection")
    description: str = Field(..., title="Content of the subsection")

    @property
    def as_str(self) -> str:
        return f"### {self.subsection_title}\n\n{self.description}".strip()


class Section(BaseModel):
    section_title: str = Field(..., title="Title of the section")
    description: str = Field(..., title="Content of the section")
    subsections: Optional[List[Subsection]] = Field(default=None, title="Titles and descriptions for each subsection of the Wikipedia page.")

    @property
    def as_str(self) -> str:
        subsections = "\n\n".join(f"### {subsection.subsection_title}\n\n{subsection.description}" for subsection in self.subsections or [])
        return f"## {self.section_title}\n\n{self.description}\n\n{subsections}".strip()


class Outline(BaseModel):
    page_title: str = Field(..., title="Title of the Wikipedia page")
    sections: List[Section] = Field(default_factory=list, title="Titles and descriptions for each section of the Wikipedia page.")

    @property
    def as_str(self) -> str:
        sections = "\n\n".join(section.as_str for section in self.sections)
        return f"# {self.page_title}\n\n{sections}".strip()


generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(Outline)

example_topic = "Impact of million-plus token context window language models on RAG"

initial_outline = generate_outline_direct.invoke({"topic": example_topic})

print(initial_outline.as_str)

gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """
I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.
Please list the as many subjects and urls as you can.
Topic of interest: {topic}
"""
)


class RelatedSubjects(BaseModel):
    topics: List[str] = Field(description="Comprehensive list of related subjects as background research.")


expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(RelatedSubjects)

related_subjects = expand_chain.ainvoke({"topic": example_topic})
print(related_subjects)


class Editor(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the editor.")
    name: str = Field(description="Name of the editor.", pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    role: str = Field(description="Role of the editor in the context of the topic.")
    description: str = Field(description="Description of the editor's focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"


class Perspectives(BaseModel):
    editors: List[Editor] = Field(
        description="Comprehensive list of editors with their roles and affiliations.",
        # Add a pydantic validation/restriction to be at most M editors
    )


gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.\nYou can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.\nWiki page outlines of related topics for inspiration:\n{examples}",
        ),
        ("user", "Topic of interest: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | long_context_llm.with_structured_output(Perspectives)
