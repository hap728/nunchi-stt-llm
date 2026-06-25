from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

REPLACEMENT_CHAR = "\ufffd"
ASCII_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9+._-]*$")
MULTISPACE_PATTERN = re.compile(r"\s{2,}")


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    text: str
    replacement_count: int


@dataclass(frozen=True, slots=True)
class TermRule:
    target: str
    pattern: re.Pattern[str]


class TranscriptNormalizer:
    def __init__(self, terms_path: str) -> None:
        self.terms_path = Path(terms_path)
        self.rules = self._load_rules(self.terms_path)
        print(
            "[STT] text normalizer initialized "
            f"terms={self.terms_path} "
            f"rules={len(self.rules)}"
        )

    def normalize(self, text: str) -> NormalizationResult:
        if not text:
            return NormalizationResult(text="", replacement_count=0)

        replacement_count = text.count(REPLACEMENT_CHAR)
        corrected = self._clean_replacement_chars(text)

        for rule in self.rules:
            corrected, count = rule.pattern.subn(rule.target, corrected)
            replacement_count += count

        corrected = MULTISPACE_PATTERN.sub(" ", corrected).strip()
        return NormalizationResult(text=corrected, replacement_count=replacement_count)

    def _clean_replacement_chars(self, text: str) -> str:
        if REPLACEMENT_CHAR not in text:
            return text
        return text.replace(REPLACEMENT_CHAR, "")

    def _load_rules(self, path: Path) -> list[TermRule]:
        if not path.is_file():
            log.warning("Domain term dictionary not found: %s", path)
            return []

        rules: list[TermRule] = []
        seen: set[tuple[str, str]] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            parts = [part.strip() for part in stripped.split("\t") if part.strip()]
            if len(parts) < 2:
                log.warning("Skipping invalid domain term line %s:%d", path, line_number)
                continue

            target = parts[0]
            for variant in sorted(set(parts[1:]), key=len, reverse=True):
                key = (target, variant)
                if key in seen or variant == target:
                    continue
                seen.add(key)
                rules.append(TermRule(target=target, pattern=self._compile_variant_pattern(variant)))

        rules.sort(key=lambda rule: len(rule.pattern.pattern), reverse=True)
        return rules

    def _compile_variant_pattern(self, variant: str) -> re.Pattern[str]:
        if ASCII_TOKEN_PATTERN.match(variant):
            return re.compile(rf"(?<![A-Za-z0-9+._-]){re.escape(variant)}(?![A-Za-z0-9+._-])")

        escaped = re.escape(variant)
        flexible_space = re.sub(r"\\\s+", r"\\s*", escaped)
        return re.compile(flexible_space, re.IGNORECASE)
