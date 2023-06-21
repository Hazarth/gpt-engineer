from enum import Enum
from typing import List

from gpt_engineer.ai import AI
from gpt_engineer.db import DBs
from gpt_engineer.fork.steps import (
    ClarificationStep,
    ExecuteEntrypoint,
    FixCode,
    GenerateCode,
    GenerateEntrypoint,
    GenerateSpec,
    GenerateUnitTests,
    ReSpec,
    RunLatest,
    RunMain,
    Step,
    StepRunner,
    UseFeedback,
)


def runner(steps: List[Step]):
    def construct(ai: AI, dbs: DBs):
        return StepRunner(ai, dbs, steps)

    return construct


class Config(str, Enum):
    DEFAULT = "default"
    BENCHMARK = "benchmark"
    SIMPLE = "simple"
    TDD = "tdd"
    TDD_PLUS = "tdd+"
    CLARIFY = "clarify"
    RESPEC = "respec"
    EXECUTE_ONLY = "execute_only"
    USE_FEEDBACK = "use_feedback"


# Different configs of what steps to run
STEPS = {
    Config.DEFAULT: runner(
        [ClarificationStep(), RunLatest(), GenerateEntrypoint(), ExecuteEntrypoint()]
    ),
    Config.BENCHMARK: runner([RunMain(), GenerateEntrypoint()]),
    Config.SIMPLE: runner([RunMain(), GenerateEntrypoint(), ExecuteEntrypoint()]),
    Config.TDD: runner(
        [
            GenerateSpec(),
            GenerateUnitTests(),
            GenerateCode(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.TDD_PLUS: runner(
        [
            GenerateSpec(),
            GenerateUnitTests(),
            GenerateCode(),
            FixCode(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.CLARIFY: runner(
        [
            ClarificationStep(),
            RunLatest(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.RESPEC: runner(
        [
            GenerateSpec(),
            ReSpec(),
            GenerateUnitTests(),
            GenerateCode(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.USE_FEEDBACK: runner(
        [UseFeedback(), GenerateEntrypoint(), ExecuteEntrypoint()]
    ),
    Config.EXECUTE_ONLY: runner([GenerateEntrypoint(), ExecuteEntrypoint()]),
}


# Future steps that can be added:
# run_tests_and_fix_files
# execute_entrypoint_and_fix_files_if_needed
