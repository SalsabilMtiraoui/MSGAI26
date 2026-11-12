"""Task scenarios: a user goal + an injectable untrusted document/email.

Each task defines a benign workspace interaction (summarize this, forward that)
whose data carries a ``{INJECTION}`` slot. The attacker fills the slot; the
judge checks the resulting :class:`Environment` state for (a) whether the
*attack goal* happened and (b) whether the *user task* still completed (stealth).

Tasks are loaded from JSON under ``tasks/data/<suite>.json`` so new scenarios
need no code.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Dict, List

from ..agent.environment import Email, Environment


@dataclass
class Task:
    id: str
    user_instruction: str
    context_kind: str                 # "email" | "document"
    context_template: str             # benign body, contains "{INJECTION}"
    payload_nl: str                   # instruction the attacker should embed
    benign_action: Dict               # {tool, args}
    _malicious_action: Dict           # {tool, args}, may contain "{SECRET}"
    attack_check: Dict
    utility_check: Dict
    context_meta: Dict = field(default_factory=dict)
    secret: str = ""
    allowed_recipients: List[str] = field(default_factory=list)
    _target_model: str = "unknown"    # set by the runner per episode

    # -- context / actions ------------------------------------------------
    def render_context(self, injection: str) -> str:
        return self.context_template.replace("{INJECTION}", injection)

    def malicious_action(self) -> Dict:
        a = copy.deepcopy(self._malicious_action)
        args = a.get("args", {})
        for k, v in list(args.items()):
            if isinstance(v, str):
                args[k] = v.replace("{SECRET}", self.secret)
        return a

    def attacker_payload(self) -> str:
        return self.payload_nl.replace("{SECRET}", self.secret)

    # -- environment construction ----------------------------------------
    def build_env(self) -> Environment:
        env = Environment(secrets={"api_key": self.secret} if self.secret else {})
        if self.context_kind == "email":
            env.inbox.append(
                Email(
                    id=self.context_meta.get("id", "m1"),
                    sender=self.context_meta.get("sender", "someone@corp.test"),
                    subject=self.context_meta.get("subject", "(no subject)"),
                    body="",  # body is supplied to the agent directly as context
                )
            )
        elif self.context_kind == "document":
            env.documents[self.context_meta.get("id", "doc1")] = ""
        return env


def _coerce(d: Dict) -> Task:
    return Task(
        id=d["id"],
        user_instruction=d["user_instruction"],
        context_kind=d["context_kind"],
        context_template=d["context_template"],
        payload_nl=d["payload_nl"],
        benign_action=d["benign_action"],
        _malicious_action=d["malicious_action"],
        attack_check=d["attack_check"],
        utility_check=d["utility_check"],
        context_meta=d.get("context_meta", {}),
        secret=d.get("secret", ""),
        allowed_recipients=d.get("allowed_recipients", []),
    )


def _data_dir():
    # Resolve from the `injecteval.tasks` package (which always has __init__.py)
    # rather than treating the data-only folder as an importable subpackage --
    # the latter can fail under some editable installs / setuptools versions.
    return resources.files("injecteval.tasks").joinpath("data")


def load_suite(name: str = "default") -> List[Task]:
    with _data_dir().joinpath(f"{name}.json").open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return [_coerce(t) for t in raw["tasks"]]


def list_suites() -> List[str]:
    out = []
    for p in _data_dir().iterdir():
        if p.name.endswith(".json"):
            out.append(p.name[:-5])
    return sorted(out)
