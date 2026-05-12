"""Output safety guard for Lab 24."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


@dataclass
class OutputDecision:
    is_safe: bool
    label: str
    reason: str
    latency_ms: float


class OutputGuard:
    def __init__(self) -> None:
        self.categories = {
            "violence": ["attack someone", "kill", "weapon", "bom", "tan cong"],
            "self_harm": ["self-harm", "suicide", "tu tu"],
            "hate": ["inferior race", "nguoi da den la", "do thai deu"],
            "hacking": ["hack wifi", "steal password", "bypass otp", "cach xam nhap"],
            "medical_misinfo": ["uống thuốc quá liều", "stop insulin", "bo insulin"],
        }

    def check(self, user_input: str, agent_response: str) -> OutputDecision:
        start = time.perf_counter()
        haystack = f"{user_input} {agent_response}".lower()
        for label, markers in self.categories.items():
            for marker in markers:
                if marker in haystack:
                    return OutputDecision(
                        is_safe=False,
                        label=label,
                        reason=f"Matched unsafe pattern `{marker}`.",
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )
        return OutputDecision(
            is_safe=True,
            label="safe",
            reason="No unsafe pattern detected.",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    async def check_async(self, user_input: str, agent_response: str) -> OutputDecision:
        return await asyncio.to_thread(self.check, user_input, agent_response)
