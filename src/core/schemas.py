from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["repository", "paper_or_documentation"]


class GeneratedModel(BaseModel):
    model_name: str
    resource_name: str
    # A code repository, or a well-documented paper or documentation page.
    resource_link: str
    source_type: SourceType
    summary: str
    rationale: str


class LiteratureReviewResult(BaseModel):
    candidates: list[GeneratedModel]
    # What the search tool reported (queries run, pages used), when it is available.
    # Written by the pipeline, never by the model.
    grounding: dict[str, Any] | None = None
    # Gemini token counts for the two calls of this stage (input, output). Same rule as grounding.
    gemini_usage: dict[str, Any] | None = None


class ModelCode(BaseModel):
    model_name: str
    code: str
    documentation: str = Field(description="Implementation decisions and limitations.")
