"""This module provides functions for processing transcribed text to generate formatted subtitles and AI-powered summaries using LangChain with LangGraph self-checking workflow."""

import os
import re
from typing import Any, Generator, Literal, Optional, Union

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from langchain_playground.llm import ChatOpenRouter

from .scrapper import is_youtube_url, scrap_youtube
from .utils import schema_to_string

load_dotenv()


# ============================================================================
# Configuration
# ============================================================================


class Config:
    """Centralized configuration for the summarization workflow."""

    # Model configuration
    ANALYSIS_MODEL = "openai/gpt-5-mini"
    QUALITY_MODEL = "openai/gpt-5-mini"

    # Quality thresholds
    MIN_QUALITY_SCORE = 90
    MAX_ITERATIONS = 2

    # Translation configuration
    ENABLE_TRANSLATION = False
    TARGET_LANGUAGE = "zh-TW"  # ISO language code (en, es, fr, de, etc.)


# ============================================================================
# Data Models
# ============================================================================


class Chapter(BaseModel):
    """Represents a single chapter in the analysis."""

    header: str = Field(description="A descriptive title for the chapter")
    summary: str = Field(description="A comprehensive summary of the chapter content")
    key_points: list[str] = Field(description="Important takeaways and insights from this chapter")


class Analysis(BaseModel):
    """Complete analysis of video content."""

    title: str = Field(description="The main title or topic of the video content")
    summary: str = Field(description="A comprehensive summary of the video content")
    takeaways: list[str] = Field(description="Key insights and actionable takeaways for the audience", min_length=3, max_length=8)
    key_facts: list[str] = Field(description="Important facts, statistics, or data points mentioned", min_length=3, max_length=6)
    chapters: list[Chapter] = Field(description="Structured breakdown of content into logical chapters")
    keywords: list[str] = Field(description="The most relevant keywords in the analysis worthy of highlighting", min_length=3, max_length=3)
    target_language: Optional[str] = Field(default=None, description="The language the content to be translated to")


class Rate(BaseModel):
    """Quality rating for a single aspect."""

    rate: Literal["Fail", "Refine", "Pass"] = Field(description="Score for the quality aspect (Fail=poor, Refine=adequate, Pass=excellent)")
    reason: str = Field(description="Reason for the score")


class Quality(BaseModel):
    """Quality assessment of the analysis."""

    completeness: Rate = Field(description="Rate for completeness: The entire transcript has been considered")
    structure: Rate = Field(description="Rate for structure: The result is in desired structures")
    grammar: Rate = Field(description="Rate for grammar: No typos, grammatical mistakes, appropriate wordings")
    no_garbage: Rate = Field(description="Rate for no_garbage: The promotional and meaningless content are removed")
    meta_language_avoidance: Rate = Field(description="Rate for meta-language avoidance: No phrases like 'This chapter introduces', 'This section covers', etc.")
    useful_keywords: Rate = Field(description="Rate for keywords: The keywords are useful for highlighting the analysis")
    correct_language: Rate = Field(description="Rate for language: Match the original language of the transcript or user requested")

    @property
    def all_aspects(self) -> list[Rate]:
        """Return all quality aspects as a list."""
        return [
            self.completeness,
            self.structure,
            self.grammar,
            self.no_garbage,
            self.meta_language_avoidance,
            self.useful_keywords,
            self.correct_language,
        ]

    @property
    def percentage_score(self) -> int:
        """Calculate percentage score based on Pass/Refine/Fail ratings."""
        pass_count = sum(1 for aspect in self.all_aspects if aspect.rate == "Pass")
        refine_count = sum(1 for aspect in self.all_aspects if aspect.rate == "Refine")
        total = len(self.all_aspects)

        # Pass = 100%, Refine = 50%, Fail = 0%
        score = (pass_count * 100 + refine_count * 50) / total
        return int(score)

    @property
    def is_acceptable(self) -> bool:
        """Check if quality score meets minimum threshold."""
        return self.percentage_score >= Config.MIN_QUALITY_SCORE


class GraphInput(BaseModel):
    """Input schema for the summarization graph."""

    transcript_or_url: str = Field(description="YouTube URL or transcript text")


class GraphState(BaseModel):
    """State schema for the summarization graph."""

    transcript: Optional[str] = None
    analysis: Optional[Analysis] = None
    quality: Optional[Quality] = None
    iteration_count: int = Field(default=0)
    is_complete: bool = Field(default=False)


class GraphOutput(BaseModel):
    """Output schema for the summarization graph."""

    analysis: Analysis
    quality: Optional[Quality] = None
    iteration_count: int
    transcript: Optional[str] = None


# ============================================================================
# Graph Nodes
# ============================================================================


class LangChainAnalysisNode:
    """Node for generating analysis using LangChain."""

    @staticmethod
    def execute(state: GraphState) -> dict:
        """Execute analysis generation."""
        llm = ChatOpenRouter(
            model=Config.ANALYSIS_MODEL,
            temperature=0,
        )

        analysis_schema_str = schema_to_string(Analysis)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are an expert content analyst. Analyze the provided transcript and create a comprehensive analysis.

Output Schema:
{analysis_schema_str}

Guidelines:
- Extract key information accurately
- Structure content into logical chapters
- Identify important facts and takeaways
- Select meaningful keywords
- Write in clear, concise language
- Avoid meta-language phrases like "This chapter introduces" or "This section covers"
""",
                ),
                ("human", "Transcript:\n{transcript}"),
            ]
        )

        structured_llm = llm.with_structured_output(Analysis)
        chain = prompt | structured_llm

        analysis = chain.invoke({"transcript": state.transcript})

        return {
            "analysis": analysis,
            "iteration_count": state.iteration_count + 1,
        }


class LangChainQualityNode:
    """Node for quality assessment using LangChain."""

    @staticmethod
    def execute(state: GraphState) -> dict:
        """Execute quality assessment."""
        llm = ChatOpenRouter(
            model=Config.QUALITY_MODEL,
            temperature=0,
        )

        quality_schema_str = schema_to_string(Quality)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are a quality assessor. Evaluate the analysis against the transcript.

Quality Schema:
{quality_schema_str}

Transcript:
{{transcript}}

Analysis:
{{analysis}}

Evaluate each aspect and provide ratings (Pass/Refine/Fail) with reasons.""",
                ),
                (
                    "human",
                    "Please assess the quality of this analysis based on the transcript provided.",
                ),
            ]
        )

        structured_llm = llm.with_structured_output(Quality)
        chain = prompt | structured_llm

        quality = chain.invoke(
            {
                "transcript": state.transcript,
                "analysis": state.analysis.model_dump_json() if state.analysis else "",
            }
        )

        return {"quality": quality, "is_complete": quality.is_acceptable}


# ============================================================================
# Graph Construction
# ============================================================================


def should_continue_langchain(state: GraphState) -> str:
    """Determine next step in LangChain workflow."""
    if state.is_complete:
        print(f"✅ LangChain workflow complete (quality: {state.quality.percentage_score if state.quality else 'None'}%)")
        return END
    elif state.quality and not state.quality.is_acceptable and state.iteration_count < Config.MAX_ITERATIONS:
        print(f"🔄 LangChain quality {state.quality.percentage_score}% below threshold {Config.MIN_QUALITY_SCORE}%, re-entering analysis (iteration {state.iteration_count + 1})")
        return "langchain_analysis"
    else:
        print(f"🔄 LangChain workflow ending (quality: {state.quality.percentage_score if state.quality else 'None'}%, iterations: {state.iteration_count})")
        return END


def create_summarization_graph() -> StateGraph:
    """Create the summarization workflow graph with conditional routing."""
    builder = StateGraph(GraphState, output_schema=GraphOutput)

    # Add nodes
    builder.add_node("langchain_analysis", LangChainAnalysisNode.execute)
    builder.add_node("langchain_quality", LangChainQualityNode.execute)

    # Add edge from START to analysis
    builder.add_edge(START, "langchain_analysis")

    # Add edge from analysis to quality
    builder.add_edge("langchain_analysis", "langchain_quality")

    # Add conditional edges from quality node
    builder.add_conditional_edges(
        "langchain_quality",
        should_continue_langchain,
        {
            "langchain_analysis": "langchain_analysis",
            END: END,
        },
    )

    return builder


def create_compiled_graph():
    """Create and compile the summarization graph."""
    return create_summarization_graph().compile()


# ============================================================================
# Public API
# ============================================================================


def summarize_video(transcript_or_url: str) -> Analysis:
    """Summarize the text using LangChain with LangGraph self-checking workflow.

    Args:
        transcript_or_url: YouTube URL or transcript text

    Returns:
        Analysis: Complete analysis of the video content
    """
    graph = create_compiled_graph()

    # Extract transcript if URL provided
    transcript = transcript_or_url
    if is_youtube_url(transcript_or_url):
        result = scrap_youtube(transcript_or_url)
        if not result.has_transcript:
            raise ValueError("Video does not have a transcript available")
        transcript = result.parsed_transcript or ""

    # Invoke with GraphState
    initial_state = GraphState(transcript=transcript)
    result: dict = graph.invoke(initial_state.model_dump())
    result: GraphOutput = GraphOutput.model_validate(result)

    print(f"🎯 Final quality score: {result.quality.percentage_score if result.quality else 'N/A'}% (after {result.iteration_count} iterations)")
    return result.analysis


def stream_summarize_video(transcript_or_url: str) -> Generator[GraphState, None, None]:
    """Stream the summarization process with progress updates using LangGraph's stream_mode='values'.

    This allows for both getting adhoc progress status updates and the final result.

    The final chunk will contain the complete graph state with the final analysis.

    Args:
        transcript_or_url: YouTube URL or transcript text

    Yields:
        GraphState: Current state of the summarization process
    """
    graph = create_compiled_graph()

    # Extract transcript if URL provided
    transcript = transcript_or_url
    if is_youtube_url(transcript_or_url):
        result = scrap_youtube(transcript_or_url)
        if not result.has_transcript:
            raise ValueError("Video does not have a transcript available")
        transcript = result.parsed_transcript or ""

    # Stream with GraphState
    initial_state = GraphState(transcript=transcript)
    for chunk in graph.stream(initial_state.model_dump(), stream_mode="values"):
        yield GraphState.model_validate(chunk)
