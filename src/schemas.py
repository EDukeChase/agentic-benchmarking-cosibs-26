from pydantic import BaseModel, Field

class GeneratedModel(BaseModel):
    model_name: str
    resource_name: str
    resource_link: str
    summary: str
    rationale: str

class LiteratureReviewResult(BaseModel):
    candidates: list[GeneratedModel]

class ModelCode(BaseModel):
    model_name: str
    code: str
    documentation: str=Field(description="Implementation decisions and limitations.")

class BenchmarkResult(BaseModel):
    model_name: str
    accuracy: float
    f1: float
    precision: float
    recall: float
    auroc: float
    brier: float
    threshold: float = 0.5
    uncertainty: float = 0.0

class ReportNarrative(BaseModel):
    summary: str
    recommendations: str

class ModelReportEntry(BaseModel):
    model_name: str
    resource_name: str
    resource_link: str
    rationale: str
    code: str
    documentation: str
    benchmark_code: str
    status: str
    accuracy: float
    f1: float
    auroc: float
    precision: float
    recall: float
    brier: float
    threshold: float = 0.5

class BenchmarkReport(BaseModel):
    entries: list[ModelReportEntry]
    summary: str
    recommendations: str
    uncertainty: float | None = None
