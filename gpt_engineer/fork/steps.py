import json
import re
import subprocess

from typing import List

from gpt_engineer.ai import AI
from gpt_engineer.chat_to_files import to_files
from gpt_engineer.db import DBs
from gpt_engineer.learning import human_input


class Step:
    step_id: str = "undefined"

    def __init__(self, name):
        self.name = name
        self.prev = None

    def __call__(self, runner: "StepRunner"):
        self.prev = runner.prev_step
        self.messages = self.run(runner.ai, runner.dbs)
        return self.messages

    def run(self, ai: AI, dbs: DBs):
        pass


class StepRunner:
    def __init__(self, ai: AI, dbs: DBs, steps: List[Step]):
        self.ai = ai
        self.dbs = dbs
        self.steps = steps
        self.prev_step = None

    def run(self):
        for step in self.steps:
            messages = step(self)
            self.dbs.logs[step.step_id] = json.dumps(messages)
            self.prev_step = step


def setup_sys_prompt(dbs):
    return (
        dbs.preprompts["generate"] + "\nUseful to know:\n" + dbs.preprompts["philosophy"]
    )


class ClarificationStep(Step):
    step_id: str = "clarification"

    def __init__(self):
        Step.__init__(self, "Clarification")

    def run(self, ai: AI, dbs: DBs):
        """
        Ask the user if they want to clarify anything
        and save the results to the workspace
        """
        messages = [ai.fsystem(dbs.preprompts["qa"])]
        user = dbs.input["main_prompt"]
        while True:
            messages = ai.next(messages, user)

            if messages[-1]["content"].strip().lower().startswith("no"):
                break

            print()
            user = input('(answer in text, or "c" to move on)\n')
            print()

            if not user or user == "c":
                break

            user += (
                "\n\n"
                "Is anything else unclear? If yes, only answer in the form:\n"
                "{remaining unclear areas} remaining questions.\n"
                "{Next question}\n"
                'If everything is sufficiently clear, only answer "no".'
            )

        print()
        return messages


class GenClarifiedCode(Step):
    step_id: str = "run_latest"

    def __init__(self):
        Step.__init__(self, "Run Latest")

    def run(self, ai: AI, dbs: DBs):
        # get the messages from previous step
        messages = self.prev.messages

        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
        ] + messages[1:]
        messages = ai.next(messages, dbs.preprompts["use_qa"])
        to_files(messages[-1]["content"], dbs.workspace)
        return messages


class Planning(Step):
    step_id: str = "planning"

    def __init__(self):
        Step.__init__(self, "Planning")

    def run(self, ai: AI, dbs: DBs):
        # get the messages from previous step
        messages = self.prev.messages

        messages = [
            ai.fsystem("You provide additional creative information about the project."),
        ] + messages[1:]
        messages = ai.next(messages, dbs.preprompts["planning"])
        return messages


class SimpleGen(Step):
    step_id: str = "run_main"

    def __init__(self):
        Step.__init__(self, "Run Main")

    def run(self, ai: AI, dbs: DBs):
        """Run the AI on the main prompt and save the results"""
        messages = ai.start(setup_sys_prompt(dbs), dbs.input["main_prompt"], self.step_id)
        to_files(messages[-1]["content"], dbs.workspace)
        return messages


class GenerateSpec(Step):
    step_id: str = "gen_spec"

    def __init__(self):
        Step.__init__(self, "Generate Specification")

    def run(self, ai: AI, dbs: DBs):
        """
        Generate a spec from the main prompt
        and clarifications and save the results to the workspace
        """
        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
            ai.fsystem(f"Instructions: {dbs.input['main_prompt']}"),
        ]

        messages = ai.next(messages, dbs.preprompts["spec"])

        dbs.memory["specification"] = messages[-1]["content"]

        return messages


class ReSpec(Step):
    step_id: str = "respec"

    def __init__(self):
        Step.__init__(self, "Regenerate Specification")

    def run(self, ai: AI, dbs: DBs):
        messages = dbs.logs[GenerateSpec.step_id]
        messages += [ai.fsystem(dbs.preprompts["respec"])]

        messages = ai.next(messages)
        messages = ai.next(
            messages,
            (
                "Based on the conversation so far,"
                "please reiterate the specification for the program. "
                "If there are things that can be improved,"
                "please incorporate the improvements. "
                "If you are satisfied with the specification, "
                "just write out the specification word by word again."
            ),
        )

        dbs.memory["specification"] = messages[-1]["content"]
        return messages


class GenerateUnitTests(Step):
    step_id: str = "gen_unit_tests"

    def __init__(self):
        Step.__init__(self, "Generate Unit Tests")

    def run(self, ai: AI, dbs: DBs):
        """
        Generate unit tests based on the specification, that should work.
        """
        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
            ai.fuser(f"Instructions: {dbs.input['main_prompt']}"),
            ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
        ]

        messages = ai.next(messages, dbs.preprompts["unit_tests"])

        dbs.memory["unit_tests"] = messages[-1]["content"]
        to_files(dbs.memory["unit_tests"], dbs.workspace)

        return messages


class GenerateCode(Step):
    step_id: str = "gen_code"

    def __init__(self):
        Step.__init__(self, "Generate Code")

    def run(self, ai: AI, dbs: DBs):
        # get the messages from previous step

        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
            ai.fuser(f"Instructions: {dbs.input['main_prompt']}"),
            ai.fuser(f"Specification:\n\n{dbs.memory['specification']}"),
            ai.fuser(f"Unit tests:\n\n{dbs.memory['unit_tests']}"),
        ]
        messages = ai.next(messages, dbs.preprompts["use_qa"])
        to_files(messages[-1]["content"], dbs.workspace)
        return messages


class ExecuteWorkspace(Step):
    step_id: str = "exec_workspace"

    def __init__(self):
        Step.__init__(self, "Execute Workspace")

    def run(self, ai: AI, dbs: DBs):
        messages = GenerateEntrypoint().run(ai, dbs)
        ExecuteEntrypoint().run(ai, dbs)
        return messages


class ExecuteEntrypoint(Step):
    step_id: str = "exec_entrypoint"

    def __init__(self):
        Step.__init__(self, "Execute Entrypoint")

    def run(self, ai: AI, dbs: DBs):
        command = dbs.workspace["run.sh"]

        print("Do you want to execute this code?")
        print()
        print(command)
        print()
        print('If yes, press enter. Otherwise, type "no"')
        print()
        if input() not in ["", "y", "yes"]:
            print("Ok, not executing the code.")
            return []
        print("Executing the code...")
        print(
            "\033[92m"  # green color
            + "Note: If it does not work as expected, please consider running the code'"
            + " in another way than above."
            + "\033[0m"
        )
        print()
        subprocess.run("bash run.sh", shell=True, cwd=dbs.workspace.path)
        return []


class GenerateEntrypoint(Step):
    step_id: str = "gen_entry_point"

    def __init__(self):
        Step.__init__(self, "Generate Entrypoint")

    def run(self, ai: AI, dbs: DBs):
        messages = ai.start(
            system=(
                "You will get information about a codebase that is currently on disk in "
                "the current folder.\n"
                "From this you will answer with code blocks that includes all the "
                "necessary unix terminal commands to "
                "a) install dependencies "
                "b) run all necessary parts of the codebase (in parallel if necessary).\n"
                "Do not install globally. Do not use sudo.\n"
                "Do not explain the code, just give the commands.\n"
                "Do not use placeholders, use example values "
                "(like . for a folder argument) if necessary.\n"
            ),
            user="Information about the codebase:\n\n" + dbs.workspace["all_output.txt"],
            step_name=self.step_id,
        )
        print()

        regex = r"```\S*\n(.+?)```"
        matches = re.finditer(regex, messages[-1]["content"], re.DOTALL)
        dbs.workspace["run.sh"] = "\n".join(match.group(1) for match in matches)
        return messages


class UseFeedback(Step):
    step_id: str = "use_feedback"

    def __init__(self):
        Step.__init__(self, "Use Feedback")

    def run(self, ai: AI, dbs: DBs):
        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
            ai.fuser(f"Instructions: {dbs.input['main_prompt']}"),
            ai.fassistant(dbs.workspace["all_output.txt"]),
            ai.fsystem(dbs.preprompts["use_feedback"]),
        ]
        messages = ai.next(messages, dbs.input["feedback"])
        to_files(messages[-1]["content"], dbs.workspace)
        return messages


class FixCode(Step):
    step_id: str = "fix_code"

    def __init__(self):
        Step.__init__(self, "Fix Code")

    def run(self, ai: AI, dbs: DBs):
        code_ouput = json.loads(dbs.logs[GenerateCode.step_id])[-1]["content"]
        messages = [
            ai.fsystem(setup_sys_prompt(dbs)),
            ai.fuser(f"Instructions: {dbs.input['main_prompt']}"),
            ai.fuser(code_ouput),
            ai.fsystem(dbs.preprompts["fix_code"]),
        ]
        messages = ai.next(messages, "Please fix any errors in the code above.")
        to_files(messages[-1]["content"], dbs.workspace)
        return messages


class HumanReview(Step):
    step_id: str = "human_review"

    def __init__(self):
        Step.__init__(self, "Human Review")

    def run(self, ai: AI, dbs: DBs):
        review = human_input()
        dbs.memory["review"] = review.to_json()  # type: ignore
        return []
