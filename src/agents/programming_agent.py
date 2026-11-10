from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.messages import HumanMessage, SystemMessage

from src.core.schemas import LiteratureReviewResult, ModelCode
from src.settings.config import LLMConfig
from src.settings.prompts import PROGRAMMING_STAGE_INSTRUCTIONS
from src.settings.study_prompt import StudyPrompt, compose
from src.utils.naming import slugify as _slugify

from .chat_model import build_chat_model


def build_programming_agent(root_dir: str, llm_config: LLMConfig = LLMConfig()):
    """An agent with file tools only, confined to ``root_dir``.

    It has no code execution and no web access: it writes code for methods 3 to 5
    and the benchmarking stage runs it.
    """
    Path(root_dir).mkdir(parents=True, exist_ok=True)
    return create_deep_agent(
        model=build_chat_model(llm_config),
        tools=[],
        backend=FilesystemBackend(root_dir=root_dir, virtual_mode=True),
    )


def run_programming_agent(agent, literature_result: LiteratureReviewResult, study_prompt: StudyPrompt):
    # the exact folder-name list
    name_mapping = "\n".join(
        f"- {c.model_name}  ->  folder name: {_slugify(c.model_name)}"
        for c in literature_result.candidates
    )
    system_prompt = compose(
        study_prompt, PROGRAMMING_STAGE_INSTRUCTIONS.format(name_mapping=name_mapping)
    )

    # the literature review is passed as a human message so the agent can read it
    literature_json = literature_result.model_dump_json(indent=2, exclude={"grounding", "gemini_usage"})
    human_message = (
        f"Implement the {len(literature_result.candidates)} methods described in this "
        f"literature review:\n{literature_json}\n\n"
        "Follow the folder naming rule exactly."
    )
    return agent.invoke({"messages": [SystemMessage(system_prompt), HumanMessage(human_message)]})


def collect_generated_models(output_dir: str) -> list[ModelCode]:
    # find path with generated models
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(
            f"Programming output directory was not created: {output_path}"
        )

    # get code and documentation file for each model
    models = []
    for model_dir in sorted(output_path.iterdir()):
        if not model_dir.is_dir():
            continue
        code_file = model_dir / "model.py"
        docs_file = model_dir / "docs.md"
        if not code_file.exists():
            print(f"Warning: skipping '{model_dir.name}': no model.py found")
            continue

        models.append(ModelCode(
            model_name=model_dir.name,
            code=code_file.read_text(),
            documentation=docs_file.read_text() if docs_file.exists() else "No documentation provided.",
        ))
    return models
