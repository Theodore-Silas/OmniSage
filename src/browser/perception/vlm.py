"""
VLM client (v4.0) — perception channel B.

Pluggable vision provider:

  qwen-vl  — Qwen-VL via DashScope OpenAI-compatible vision endpoint (default)
  openai   — GPT-4o / GPT-4.1 vision
  local    — local VLM via transformers (offline, no API cost)
  none     — disabled; the agent falls back to channel A (DOM only)

The client is defensive: any provider failure degrades to ``None`` so the
agent can continue on channel A instead of crashing.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from openai import AsyncOpenAI

from src.browser.driver.base import InteractiveElement
from src.config import VLMConfig


@dataclass
class VLMResult:
    """Structured perception result from a vision model."""
    element_id: int = -1
    action: str = ""
    reason: str = ""
    page_summary: str = ""
    raw: str = ""


_PROMPT_TEMPLATE = """You are the vision module of a web automation agent. A screenshot is attached with RED numbered boxes marking interactive elements.

Task: {task}

Analyze the screenshot and return ONLY a JSON object (no markdown, no prose) with these fields:
- "element_id": integer id of the element to interact with, or -1 if none matches
- "action": one of "click", "type", "scroll", "hover", "select", "none"
- "reason": one short sentence explaining your choice
- "page_summary": one short sentence describing what the page shows

Example: {{"element_id": 12, "action": "click", "reason": "The search button matches the task", "page_summary": "A product search page"}}"""


class VLMClient:
    """Vision-language model client with graceful degradation."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self._client: Optional[AsyncOpenAI] = None
        self._local = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        return self._client

    # -- main entry ----------------------------------------------------

    async def perceive(
        self,
        screenshot_bytes: bytes,
        som_bytes: Optional[bytes],
        task: str,
        elements: Optional[List[InteractiveElement]] = None,
    ) -> Optional[VLMResult]:
        """Perceive the current screen. Returns None when disabled or failed."""
        if not self.enabled:
            return None

        image_bytes = som_bytes or screenshot_bytes
        try:
            if self.config.provider in ("openai", "qwen-vl"):
                return await self._perceive_api(image_bytes, task)
            if self.config.provider == "local":
                return await self._perceive_local(image_bytes, task)
        except Exception as e:
            return VLMResult(reason=f"VLM unavailable: {str(e)[:200]}")
        return None

    # -- providers ------------------------------------------------------

    async def _perceive_api(self, image_bytes: bytes, task: str) -> Optional[VLMResult]:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt = _PROMPT_TEMPLATE.format(task=task)
        resp = await self._get_client().chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        return self._parse(raw)

    async def _perceive_local(self, image_bytes: bytes, task: str) -> Optional[VLMResult]:
        """Local VLM via transformers (lazy-loaded). Heavy; CPU-only is slow."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if self._local is None:
                model_id = self.config.local_model
                processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id, torch_dtype=torch.float32, trust_remote_code=True
                )
                model.eval()
                self._local = (model, processor)

            model, processor = self._local
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            prompt = _PROMPT_TEMPLATE.format(task=task)
            inputs = processor(text=prompt, images=img, return_tensors="pt")
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=256)
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
            return self._parse(raw)
        except Exception as e:
            return VLMResult(reason=f"Local VLM unavailable: {str(e)[:200]}")

    # -- parsing ---------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> VLMResult:
        res = VLMResult(raw=raw)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            res.page_summary = raw.strip()[:200]
            return res
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            res.page_summary = raw.strip()[:200]
            return res

        try:
            res.element_id = int(data.get("element_id", -1))
        except (TypeError, ValueError):
            res.element_id = -1
        res.action = str(data.get("action", "")).strip()
        res.reason = str(data.get("reason", "")).strip()
        res.page_summary = str(data.get("page_summary", "")).strip()
        return res
