"""Input guardrails for Lab 24."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass

from src.lab24_common import is_domain_query, strip_accents


@dataclass
class GuardDecision:
    allowed: bool
    reason: str
    latency_ms: float


class InputGuard:
    def __init__(self) -> None:
        self.patterns = {
            "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE),
            "phone": re.compile(r"\b(?:\+84|0)(?:\d[ .-]?){8,10}\b"),
            "cccd": re.compile(r"\b\d{12}\b"),
            "tax_id": re.compile(r"\b\d{10,13}\b"),
        }
        self.injection_markers = [
            "ignore previous instructions",
            "jailbreak",
            "pretend you are dan",
            "roleplay",
            "decode this base64",
            "bypass safety",
            "khong can tuan theo",
        ]

    def sanitize(self, text: str) -> tuple[str, dict[str, int]]:
        start = time.perf_counter()
        counts = {}
        sanitized = text
        for name, pattern in self.patterns.items():
            matches = pattern.findall(sanitized)
            if matches:
                counts[name] = len(matches)
                sanitized = pattern.sub(f"[REDACTED_{name.upper()}]", sanitized)
        counts["latency_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return sanitized, counts

    async def sanitize_async(self, text: str) -> tuple[str, dict[str, int]]:
        return await asyncio.to_thread(self.sanitize, text)

    def check_topic(self, text: str) -> GuardDecision:
        start = time.perf_counter()
        allowed = is_domain_query(text)
        reason = "in_scope" if allowed else "Câu hỏi nằm ngoài phạm vi pháp lý và bảo vệ dữ liệu của hệ thống."
        return GuardDecision(allowed=allowed, reason=reason, latency_ms=(time.perf_counter() - start) * 1000)

    async def check_topic_async(self, text: str) -> GuardDecision:
        return await asyncio.to_thread(self.check_topic, text)

    def detect_injection(self, text: str) -> GuardDecision:
        start = time.perf_counter()
        normalized = strip_accents(text)
        matched = [marker for marker in self.injection_markers if marker in normalized]
        allowed = not matched
        reason = "clean" if allowed else f"Prompt injection marker detected: {matched[0]}"
        return GuardDecision(allowed=allowed, reason=reason, latency_ms=(time.perf_counter() - start) * 1000)

    async def detect_injection_async(self, text: str) -> GuardDecision:
        return await asyncio.to_thread(self.detect_injection, text)


def graceful_refusal() -> str:
    return "Mình chỉ hỗ trợ các câu hỏi thuộc phạm vi pháp luật, an ninh mạng và bảo vệ dữ liệu cá nhân trong bộ tài liệu hiện có."
