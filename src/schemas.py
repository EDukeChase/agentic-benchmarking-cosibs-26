from pydantic import BaseModel, Field

class GeneratedModel(BaseModel):
    model_name: str
    resource_name: str
    resource_link: str
    summary: str
    rationale: str

class ModelCode(BaseModel):
    model_name: str
    code: str
    documentation: str=Field(description="Implementation decisions and limitations.")

class BenchmarkResult(BaseModel):
    model_name: str
    accuracy: float
    f1: float
    auroc: float

class BenchmarkReport(BaseModel):
    results: list[BenchmarkResult]
    summary: str
    recommendations: str