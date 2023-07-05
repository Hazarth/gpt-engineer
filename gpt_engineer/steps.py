from enum import Enum
from typing import List

from gpt_engineer.ai import AI
from gpt_engineer.db import DBs
from gpt_engineer.fork.steps import (
    ClarificationStep,
    ExecuteEntrypoint,
    FixCode,
    GenClarifiedCode,
    GenerateCode,
    GenerateEntrypoint,
    GenerateSpec,
    GenerateUnitTests,
    HumanReview,
    ReSpec,
    SimpleGen,
    Step,
    StepRunner,
    UseFeedback,
)


def setup_sys_prompt(dbs):
    return dbs.identity["generate"] + "\nUseful to know:\n" + dbs.identity["philosophy"]


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
    EVALUATE = "evaluate"
    USE_FEEDBACK = "use_feedback"


# Different configs of what steps to run
STEPS = {
    Config.DEFAULT: runner(
        [
            ClarificationStep(),
            GenClarifiedCode(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.BENCHMARK: runner([SimpleGen(), GenerateEntrypoint()]),
    Config.SIMPLE: runner([SimpleGen(), GenerateEntrypoint(), ExecuteEntrypoint()]),
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
            GenClarifiedCode(),
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
            FixCode(),
            GenerateEntrypoint(),
            ExecuteEntrypoint(),
        ]
    ),
    Config.USE_FEEDBACK: runner(
        [UseFeedback(), GenerateEntrypoint(), ExecuteEntrypoint()]
    ),
    Config.EXECUTE_ONLY: runner([GenerateEntrypoint(), ExecuteEntrypoint()]),
    Config.EVALUATE: runner([ExecuteEntrypoint(), HumanReview()]),
}


# Future steps that can be added:
# run_tests_and_fix_files
# execute_entrypoint_and_fix_files_if_it_results_in_error
