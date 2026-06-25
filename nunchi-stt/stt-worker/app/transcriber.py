from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.transcribe import Segment
from faster_whisper.utils import download_model

from .config import Settings

log = logging.getLogger(__name__)

# Keep the base decode as close to the raw audio as possible.
# Free-form prompt guidance is a common source of plausible hallucinations.
INITIAL_PROMPT: str | None = None

# Conservative decode thresholds to drop low-confidence silent tails
# before they turn into repeated or caption-like hallucinations.
LOG_PROB_THRESHOLD = -0.6
NO_SPEECH_THRESHOLD = 0.55
VAD_MIN_SILENCE_DURATION_MS = 400
VAD_SPEECH_PAD_MS = 200
HALLUCINATION_SILENCE_THRESHOLD = 1.0
PRIMARY_BEAM_SIZE = 5
PRIMARY_BEST_OF = 1
RERUN_MARGIN_SEC = 0.4
RERUN_MIN_IMPROVEMENT = 0.2
RERUN_LOG_PROB_THRESHOLD = -0.75
RERUN_NO_SPEECH_THRESHOLD = 0.4
RERUN_VAD_MIN_SILENCE_DURATION_MS = 300
RERUN_VAD_SPEECH_PAD_MS = 150
RERUN_HALLUCINATION_SILENCE_THRESHOLD = 0.8

HOTWORDS: str | None = None

CHUNK_TRANSCRIBE_ENABLED = True
CHUNK_MIN_DURATION_SEC = 20 * 60
CHUNK_LENGTH_SEC = 15 * 60
CHUNK_OVERLAP_SEC = 10


WEEKDAY_TOKENS = (
    "\uc6d4\uc694\uc77c",
    "\ud654\uc694\uc77c",
    "\uc218\uc694\uc77c",
    "\ubaa9\uc694\uc77c",
    "\uae08\uc694\uc77c",
    "\ud1a0\uc694\uc77c",
    "\uc77c\uc694\uc77c",
)
WEEKDAY_SET = set(WEEKDAY_TOKENS)
SEPARATOR_PATTERN = re.compile(r"[\s,./-]+")
TOKEN_SPLIT_PATTERN = re.compile(r"(\s+)")
WHITESPACE_PATTERN = re.compile(r"\s{2,}")
NON_WORD_PATTERN = re.compile(r"[^\w\uac00-\ud7a3]")
HANGUL_PATTERN = re.compile(r"[\uac00-\ud7a3]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")
WORD_PATTERN = re.compile(r"[A-Za-z0-9\uac00-\ud7a3]+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
SAFE_REPEAT_WORDS = {
    "\ub124",
    "\uc74c",
    "\uc544",
    "\uadf8",
    "\uc800\ud76c",
    "\uc800\ud76c\uac00",
    "\uc81c\uac00",
    "\uadf8\ub9ac\uace0",
}


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    source: str = "primary"


class Transcriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type
        self.batch_size = max(1, settings.whisper_batch_size)
        self.enable_loudnorm = settings.ffmpeg_enable_loudnorm
        self.model_dir = Path(self.settings.whisper_model_dir) / self.settings.whisper_model_size
        self.model_path = self._ensure_model_files()
        self.model = self._create_model(self.device, self.compute_type)
        self.pipeline = self._create_pipeline()
        print(
            "[STT] model initialized "
            f"model={self.settings.whisper_model_size} "
            f"device={self.device} "
            f"compute_type={self.compute_type} "
            f"pipeline={self._pipeline_name()} "
            f"batch_size={self.batch_size} "
            f"loudnorm={self.enable_loudnorm}"
        )

    def transcribe(self, audio_path: Path) -> str:
        started_at = time.perf_counter()
        preprocessed_path = self._preprocess_audio(audio_path)
        target_path = preprocessed_path if preprocessed_path is not None else audio_path

        print(
            "[STT] transcribe start "
            f"path={target_path.name} "
            f"device={self.device} "
            f"compute_type={self.compute_type} "
            f"pipeline={self._pipeline_name()}"
        )

        try:
            return self._run_chunked_transcription_if_needed(target_path)
        finally:
            elapsed = time.perf_counter() - started_at
            print(
                "[STT] transcribe finished "
                f"path={target_path.name} "
                f"elapsed={elapsed:.1f}s "
                f"device={self.device} "
                f"compute_type={self.compute_type} "
                f"pipeline={self._pipeline_name()}"
            )
            if preprocessed_path is not None and preprocessed_path.exists():
                preprocessed_path.unlink(missing_ok=True)

    def _run_with_fallbacks(self, audio_path: Path) -> str:
        try:
            return self._run_transcription(audio_path)
        except RuntimeError as error:
            if not self._is_cuda_oom(error) or self.device != "cuda":
                raise

            for device, compute_type in self._fallback_plan():
                log.warning("CUDA OOM with %s. Retrying with %s/%s.", self.compute_type, device, compute_type)
                self._reload_model(device=device, compute_type=compute_type)
                try:
                    return self._run_transcription(audio_path)
                except RuntimeError as retry_error:
                    if not self._is_cuda_oom(retry_error) or device != "cuda":
                        raise

            raise

    def _run_transcription(self, audio_path: Path) -> str:
        transcribe_kwargs = self._build_primary_transcribe_kwargs()
        text = self._transcribe_to_text(audio_path, transcribe_kwargs, use_pipeline=self.pipeline is not None)

        if self._looks_like_non_korean_output(text):
            log.warning("Detected likely non-Korean output on primary decode. Retrying with single decoder path.")
            if self.pipeline is not None:
                retry_text = self._transcribe_to_text(audio_path, transcribe_kwargs, use_pipeline=False)
                if not self._looks_like_non_korean_output(retry_text):
                    return retry_text
            raise RuntimeError("KOREAN_LANGUAGE_SANITY_CHECK_FAILED")

        return text

    def _build_primary_transcribe_kwargs(self) -> dict[str, object]:
        return {
            "language": "ko",
            "task": "transcribe",
            "beam_size": PRIMARY_BEAM_SIZE,
            "best_of": PRIMARY_BEST_OF,
            "temperature": 0.0,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 3,
            "compression_ratio_threshold": 2.0,
            "log_prob_threshold": LOG_PROB_THRESHOLD,
            "no_speech_threshold": NO_SPEECH_THRESHOLD,
            "vad_filter": True,
            "vad_parameters": {
                "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
                "speech_pad_ms": VAD_SPEECH_PAD_MS,
            },
            "condition_on_previous_text": False,
            "prompt_reset_on_temperature": 0.0,
            "initial_prompt": INITIAL_PROMPT,
            "hotwords": HOTWORDS,
            "word_timestamps": False,
            "without_timestamps": True,
            "hallucination_silence_threshold": HALLUCINATION_SILENCE_THRESHOLD,
        }

    def _build_rerun_transcribe_kwargs(self) -> dict[str, object]:
        kwargs = self._build_primary_transcribe_kwargs()
        kwargs.update(
            {
                "log_prob_threshold": RERUN_LOG_PROB_THRESHOLD,
                "no_speech_threshold": RERUN_NO_SPEECH_THRESHOLD,
                "vad_parameters": {
                    "min_silence_duration_ms": RERUN_VAD_MIN_SILENCE_DURATION_MS,
                    "speech_pad_ms": RERUN_VAD_SPEECH_PAD_MS,
                },
                "initial_prompt": None,
                "hotwords": None,
                "hallucination_silence_threshold": RERUN_HALLUCINATION_SILENCE_THRESHOLD,
            }
        )
        return kwargs

    def _transcribe_to_text(self, audio_path: Path, transcribe_kwargs: dict[str, object], use_pipeline: bool) -> str:
        segments = self._transcribe_segments(audio_path, transcribe_kwargs, use_pipeline)
        filtered_segments = self._filter_segments(segments)
        return self._post_process_text(self._segments_to_text(filtered_segments))

    def _transcribe_segments(
            self,
            audio_path: Path,
            transcribe_kwargs: dict[str, object],
            use_pipeline: bool,
    ) -> list[TranscriptSegment]:
        segments = self._decode_segments(
            audio_path,
            transcribe_kwargs,
            use_pipeline=use_pipeline,
            source="primary" if use_pipeline else "single",
        )
        return self._rerun_suspicious_segments(audio_path, segments)

    def _decode_segments(
            self,
            audio_path: Path,
            transcribe_kwargs: dict[str, object],
            use_pipeline: bool,
            source: str,
    ) -> list[TranscriptSegment]:
        if use_pipeline:
            raw_segments, _info = self.pipeline.transcribe(
                str(audio_path),
                batch_size=self.batch_size,
                **transcribe_kwargs,
            )
        else:
            raw_segments, _info = self.model.transcribe(str(audio_path), **transcribe_kwargs)

        decoded: list[TranscriptSegment] = []
        for segment in raw_segments:
            text = segment.text.strip()
            if not text:
                continue

            decoded.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    avg_logprob=float(segment.avg_logprob),
                    compression_ratio=float(segment.compression_ratio),
                    no_speech_prob=float(segment.no_speech_prob),
                    source=source,
                )
            )

        return decoded

    def _create_model(self, device: str, compute_type: str) -> WhisperModel:
        return WhisperModel(
            self.model_path,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )

    def _ensure_model_files(self) -> str:
        print(
            "[STT] ensuring model files "
            f"model={self.settings.whisper_model_size} "
            f"dir={self.model_dir}"
        )
        started_at = time.perf_counter()

        if self._has_complete_model_dir(self.model_dir):
            model_path = str(self.model_dir)
            source = "local-dir"
        else:
            self._reset_partial_model_dir(self.model_dir)
            model_path: str | None = None
            source: str | None = None

            try:
                local_model_path = download_model(
                    self.settings.whisper_model_size,
                    output_dir=str(self.model_dir),
                    local_files_only=True,
                )
                if self._has_complete_model_dir(Path(local_model_path)):
                    model_path = local_model_path
                    source = "local-cache"
            except Exception:
                model_path = None

            if model_path is None:
                self._reset_partial_model_dir(self.model_dir)
                model_path = download_model(
                    self.settings.whisper_model_size,
                    output_dir=str(self.model_dir),
                )
                source = "huggingface"

            if not self._has_complete_model_dir(Path(model_path)):
                self._reset_partial_model_dir(Path(model_path))
                raise RuntimeError(
                    f"WHISPER_MODEL_DOWNLOAD_INCOMPLETE: {self.settings.whisper_model_size}"
                )

        print(
            "[STT] model files ready "
            f"model={self.settings.whisper_model_size} "
            f"path={model_path} "
            f"source={source} "
            f"elapsed={time.perf_counter() - started_at:.1f}s"
        )
        return model_path

    def _create_pipeline(self) -> BatchedInferencePipeline | None:
        if self.device != "cuda":
            return None
        return BatchedInferencePipeline(model=self.model)

    @staticmethod
    def _has_complete_model_dir(model_dir: Path) -> bool:
        required_files = (
            "config.json",
            "model.bin",
            "tokenizer.json",
        )
        has_required_files = all((model_dir / file_name).is_file() for file_name in required_files)
        has_vocabulary = (model_dir / "vocabulary.txt").is_file() or (model_dir / "vocabulary.json").is_file()
        return model_dir.is_dir() and has_required_files and has_vocabulary

    @staticmethod
    def _reset_partial_model_dir(model_dir: Path) -> None:
        if model_dir.exists():
            shutil.rmtree(model_dir, ignore_errors=True)

    def _pipeline_name(self) -> str:
        return "batched" if self.pipeline is not None else "single"

    def _reload_model(self, device: str, compute_type: str) -> None:
        self.device = device
        self.compute_type = compute_type
        self.model = self._create_model(device, compute_type)
        self.pipeline = self._create_pipeline()
        print(
            "[STT] model reloaded "
            f"device={self.device} "
            f"compute_type={self.compute_type} "
            f"pipeline={self._pipeline_name()}"
        )

    def _fallback_plan(self) -> list[tuple[str, str]]:
        current = (self.device, self.compute_type)
        ordered = [
            ("cuda", "int8"),
        ]
        return [candidate for candidate in ordered if candidate != current]

    @staticmethod
    def _is_cuda_oom(error: RuntimeError) -> bool:
        message = str(error).lower()
        return "cuda" in message and "out of memory" in message

    @staticmethod
    def _segments_to_text(segments: list[TranscriptSegment]) -> str:
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()

    def _rerun_suspicious_segments(self, audio_path: Path, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        if not segments:
            return segments

        rerun_kwargs = self._build_rerun_transcribe_kwargs()
        refined: list[TranscriptSegment] = []

        for index, segment in enumerate(segments):
            if not self._should_rerun_segment(segment):
                refined.append(segment)
                continue

            replacement = self._rerun_single_segment(audio_path, segments, index, rerun_kwargs)
            if replacement is None:
                refined.append(segment)
                continue

            refined.append(replacement)

        return refined

    def _should_rerun_segment(self, segment: TranscriptSegment) -> bool:
        text = segment.text.strip()
        if not text:
            return False

        low_confidence = (
                segment.avg_logprob < -0.8
                or segment.no_speech_prob > 0.55
                or segment.compression_ratio > 1.55
        )

        if self._extract_weekday_only_tokens(text):
            return True

        if self._looks_like_keyword_list(text, low_confidence):
            return True

        if self._has_suspicious_keyword_burst(text, low_confidence):
            return True

        if self._is_generic_repeat_segment(text, low_confidence, repeat_count=2):
            return True

        return low_confidence and len(self._tokenize_words(text)) >= 6

    def _rerun_single_segment(
            self,
            audio_path: Path,
            segments: list[TranscriptSegment],
            index: int,
            transcribe_kwargs: dict[str, object],
    ) -> TranscriptSegment | None:
        original = segments[index]
        clip_start = max(0.0, original.start - RERUN_MARGIN_SEC)
        clip_end = max(clip_start + 0.2, original.end + RERUN_MARGIN_SEC)

        try:
            with TemporaryDirectory(prefix="nunchi-rerun-") as temp_dir:
                clip_path = Path(temp_dir) / f"segment_{index}.wav"
                self._extract_audio_clip(audio_path, clip_path, clip_start, clip_end)
                rerun_segments = self._decode_segments(
                    clip_path,
                    transcribe_kwargs,
                    use_pipeline=False,
                    source="rerun",
                )
        except RuntimeError as error:
            log.warning(
                "Rerun failed for segment index=%d start=%.2f end=%.2f: %s",
                index,
                original.start,
                original.end,
                error,
            )
            return None

        if not rerun_segments:
            return None

        shifted = [
            TranscriptSegment(
                start=clip_start + segment.start,
                end=clip_start + segment.end,
                text=segment.text,
                avg_logprob=segment.avg_logprob,
                compression_ratio=segment.compression_ratio,
                no_speech_prob=segment.no_speech_prob,
                source=segment.source,
            )
            for segment in rerun_segments
        ]

        core_segments = [
            segment
            for segment in shifted
            if self._segment_center_within(segment, original.start, original.end)
        ]
        if not core_segments:
            core_segments = [
                segment
                for segment in shifted
                if self._segment_overlaps_interval(segment, original.start, original.end)
            ]
        if not core_segments:
            return None

        candidate_text = self._segments_to_text(core_segments)
        candidate_text = self._trim_overlap_against_neighbors(
            candidate_text,
            previous_text=segments[index - 1].text if index > 0 else "",
            next_text=segments[index + 1].text if index + 1 < len(segments) else "",
        )
        if not candidate_text:
            return None

        candidate = TranscriptSegment(
            start=original.start,
            end=original.end,
            text=candidate_text,
            avg_logprob=sum(segment.avg_logprob for segment in core_segments) / len(core_segments),
            compression_ratio=sum(segment.compression_ratio for segment in core_segments) / len(core_segments),
            no_speech_prob=sum(segment.no_speech_prob for segment in core_segments) / len(core_segments),
            source="rerun",
        )

        original_score = self._segment_quality_score(original)
        candidate_score = self._segment_quality_score(candidate)
        if candidate_score < original_score + RERUN_MIN_IMPROVEMENT:
            return None

        log.info(
            "Replaced suspicious segment start=%.2f end=%.2f score=%.3f->%.3f",
            original.start,
            original.end,
            original_score,
            candidate_score,
        )
        return candidate

    @staticmethod
    def _segment_center_within(segment: TranscriptSegment, start: float, end: float) -> bool:
        midpoint = (segment.start + segment.end) / 2
        return start <= midpoint <= end

    @staticmethod
    def _segment_overlaps_interval(segment: TranscriptSegment, start: float, end: float) -> bool:
        return segment.end > start and segment.start < end

    def _segment_quality_score(self, segment: TranscriptSegment) -> float:
        text = segment.text.strip()
        if not text:
            return -999.0

        tokens = self._tokenize_words(text)
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0.0
        repeat_penalty = max(0.0, 0.65 - unique_ratio) * 3.0
        score = segment.avg_logprob * 4.0
        score -= max(0.0, segment.no_speech_prob - 0.25) * 3.5
        score -= max(0.0, segment.compression_ratio - 1.2) * 2.0
        score -= repeat_penalty

        if self._looks_like_keyword_list(text, low_confidence=True):
            score -= 2.5
        if self._has_suspicious_keyword_burst(text, low_confidence=True):
            score -= 2.0
        if self._extract_weekday_only_tokens(text):
            score -= 2.5

        return score

    def _trim_overlap_against_neighbors(self, text: str, previous_text: str, next_text: str) -> str:
        trimmed = self._drop_leading_overlap(text, previous_text)
        trimmed = self._drop_trailing_overlap(trimmed, next_text)
        return trimmed.strip()

    def _drop_leading_overlap(self, candidate_text: str, previous_text: str) -> str:
        overlap_size = self._token_overlap_suffix_prefix(previous_text, candidate_text)
        if overlap_size < 2:
            return candidate_text
        return " ".join(candidate_text.split()[overlap_size:]).strip()

    def _drop_trailing_overlap(self, candidate_text: str, next_text: str) -> str:
        overlap_size = self._token_overlap_suffix_prefix(candidate_text, next_text)
        if overlap_size < 2:
            return candidate_text
        tokens = candidate_text.split()
        return " ".join(tokens[:-overlap_size]).strip()

    def _token_overlap_suffix_prefix(self, left_text: str, right_text: str, max_tokens: int = 8) -> int:
        left_tokens = [self._normalize_token(token) for token in left_text.split() if self._normalize_token(token)]
        right_tokens = [self._normalize_token(token) for token in right_text.split() if self._normalize_token(token)]
        if not left_tokens or not right_tokens:
            return 0

        limit = min(max_tokens, len(left_tokens), len(right_tokens))
        for overlap_size in range(limit, 1, -1):
            if left_tokens[-overlap_size:] == right_tokens[:overlap_size]:
                return overlap_size
        return 0

    @staticmethod
    def _normalize_token(token: str) -> str:
        return NON_WORD_PATTERN.sub("", token).lower()

    def _filter_segments(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        filtered: list[TranscriptSegment] = []
        previous_norm = ""
        repeat_count = 0

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue

            norm = self._normalize_repeat_text(text)
            if norm and norm == previous_norm:
                repeat_count += 1
            else:
                previous_norm = norm
                repeat_count = 1

            if self._should_drop_segment(segment, text, repeat_count):
                log.warning(
                    "Dropping low-confidence suspicious segment: text=%r avg_logprob=%.3f compression_ratio=%.3f no_speech_prob=%.3f repeat_count=%d",
                    text,
                    segment.avg_logprob,
                    segment.compression_ratio,
                    segment.no_speech_prob,
                    repeat_count,
                )
                continue

            filtered.append(segment)

        return filtered

    def _should_drop_segment(self, segment: TranscriptSegment, text: str, repeat_count: int) -> bool:
        weekday_tokens = self._extract_weekday_only_tokens(text)
        low_confidence = (
                segment.avg_logprob < -0.55
                or segment.no_speech_prob > 0.45
                or segment.compression_ratio > 1.45
        )

        if weekday_tokens:
            unique_weekdays = set(weekday_tokens)

            if len(weekday_tokens) >= 3:
                return True

            if len(unique_weekdays) == 1 and repeat_count >= 3:
                return True

            if len(unique_weekdays) == 1 and low_confidence:
                return True

        if self._is_generic_repeat_segment(text, low_confidence, repeat_count):
            return True

        if self._looks_like_keyword_list(text, low_confidence):
            return True

        if self._has_suspicious_keyword_burst(text, low_confidence):
            return True

        return False

    def _extract_weekday_only_tokens(self, text: str) -> list[str]:
        compact = SEPARATOR_PATTERN.sub("", text)
        if not compact:
            return []

        tokens: list[str] = []
        index = 0
        while index < len(compact):
            matched = next((token for token in WEEKDAY_TOKENS if compact.startswith(token, index)), None)
            if matched is None:
                return []
            tokens.append(matched)
            index += len(matched)

        return tokens

    @staticmethod
    def _normalize_repeat_text(text: str) -> str:
        return SEPARATOR_PATTERN.sub("", text.strip())

    def _post_process_text(self, text: str) -> str:
        cleaned = text
        for weekday in WEEKDAY_TOKENS:
            cleaned = re.sub(
                rf"({re.escape(weekday)})(?:[\s,./-]*{re.escape(weekday)})+",
                r"\1",
                cleaned,
            )
        cleaned = self._remove_suspicious_sentences(cleaned)
        cleaned = self._collapse_repeated_weekday_tokens(cleaned)
        cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def _looks_like_non_korean_output(self, text: str) -> bool:
        compact = text.strip()
        if len(compact) < 80:
            return False

        hangul_count = len(HANGUL_PATTERN.findall(compact))
        latin_count = len(LATIN_PATTERN.findall(compact))
        alpha_count = hangul_count + latin_count

        if alpha_count < 40:
            return False

        hangul_ratio = hangul_count / alpha_count
        latin_ratio = latin_count / alpha_count
        return hangul_ratio < 0.15 and latin_ratio > 0.6

    def _collapse_repeated_weekday_tokens(self, text: str) -> str:
        parts = TOKEN_SPLIT_PATTERN.split(text)
        collapsed: list[str] = []
        previous_weekday: str | None = None

        for part in parts:
            normalized = NON_WORD_PATTERN.sub("", part)
            if normalized in WEEKDAY_SET:
                if normalized == previous_weekday:
                    continue
                previous_weekday = normalized
            elif part.strip():
                previous_weekday = None

            collapsed.append(part)

        return "".join(collapsed)

    def _is_generic_repeat_segment(self, text: str, low_confidence: bool, repeat_count: int) -> bool:
        norm = self._normalize_repeat_text(text)
        if not norm:
            return False

        if repeat_count >= 3 and len(norm) >= 8:
            return True

        return repeat_count >= 2 and low_confidence and len(norm) >= 12

    def _looks_like_keyword_list(self, text: str, low_confidence: bool) -> bool:
        tokens = [token for token in self._tokenize_words(text) if len(token) >= 2]
        if len(tokens) < 10:
            return False

        prefix_counts = Counter(token[0] for token in tokens)
        _dominant_prefix, dominant_prefix_count = prefix_counts.most_common(1)[0]
        dominant_prefix_ratio = dominant_prefix_count / len(tokens)
        short_ratio = sum(1 for token in tokens if len(token) <= 4) / len(tokens)
        comma_like_count = len(re.findall(r"[,/]", text))

        if dominant_prefix_ratio >= 0.75 and short_ratio >= 0.75:
            return True

        return dominant_prefix_ratio >= 0.6 and short_ratio >= 0.75 and low_confidence and comma_like_count >= 4

    def _has_suspicious_keyword_burst(self, text: str, low_confidence: bool) -> bool:
        tokens = [token for token in self._tokenize_words(text) if len(token) >= 2]
        if len(tokens) < 8:
            return False

        token_counts = Counter(tokens)
        dominant_token, dominant_count = token_counts.most_common(1)[0]
        if dominant_token in SAFE_REPEAT_WORDS:
            return False

        dominant_ratio = dominant_count / len(tokens)
        return dominant_count >= 4 and dominant_ratio >= 0.35 and low_confidence

    def _remove_suspicious_sentences(self, text: str) -> str:
        chunks = SENTENCE_SPLIT_PATTERN.split(text)
        if len(chunks) == 1:
            return text

        kept: list[str] = []
        for chunk in chunks:
            stripped = chunk.strip()
            if not stripped:
                continue

            if self._looks_like_keyword_list(stripped, low_confidence=False):
                continue

            if self._has_suspicious_keyword_burst(stripped, low_confidence=False):
                continue

            normalized = self._normalize_repeat_text(stripped)
            if kept and normalized and normalized == self._normalize_repeat_text(kept[-1]):
                continue

            kept.append(stripped)

        return " ".join(kept) if kept else text

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        return [match.group(0) for match in WORD_PATTERN.finditer(text)]

    def _run_chunked_transcription_if_needed(self, audio_path: Path) -> str:
        if not CHUNK_TRANSCRIBE_ENABLED:
            return self._run_with_fallbacks(audio_path)

        duration_sec = self._get_audio_duration_sec(audio_path)
        if duration_sec <= 0 or duration_sec < CHUNK_MIN_DURATION_SEC:
            return self._run_with_fallbacks(audio_path)

        print(
            "[STT] chunked transcription start "
            f"path={audio_path.name} "
            f"duration={duration_sec:.1f}s "
            f"chunk={CHUNK_LENGTH_SEC}s "
            f"overlap={CHUNK_OVERLAP_SEC}s"
        )

        chunk_texts: list[str] = []
        stride_sec = CHUNK_LENGTH_SEC - CHUNK_OVERLAP_SEC

        with TemporaryDirectory(prefix="stt_chunks_") as temp_dir:
            temp_path = Path(temp_dir)
            start_sec = 0.0
            chunk_index = 1

            while start_sec < duration_sec:
                end_sec = min(duration_sec, start_sec + CHUNK_LENGTH_SEC)
                chunk_path = temp_path / f"{audio_path.stem}_chunk_{chunk_index:04d}.wav"

                self._extract_audio_clip(audio_path, chunk_path, start_sec, end_sec)

                print(
                    "[STT] chunk start "
                    f"index={chunk_index} "
                    f"start={start_sec:.1f}s "
                    f"end={end_sec:.1f}s"
                )

                chunk_text = self._run_with_fallbacks(chunk_path)
                if chunk_text.strip():
                    chunk_texts.append(chunk_text.strip())

                print(
                    "[STT] chunk finished "
                    f"index={chunk_index} "
                    f"text_chars={len(chunk_text.replace(' ', ''))}"
                )

                if end_sec >= duration_sec:
                    break

                start_sec += stride_sec
                chunk_index += 1

        merged = self._merge_chunk_texts(chunk_texts)

        print(
            "[STT] chunked transcription finished "
            f"path={audio_path.name} "
            f"chunks={len(chunk_texts)} "
            f"text_chars={len(merged.replace(' ', ''))}"
        )

        return merged

    def _get_audio_duration_sec(self, audio_path: Path) -> float:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def _merge_chunk_texts(self, chunk_texts: list[str]) -> str:
        merged = ""

        for chunk_text in chunk_texts:
            merged = self._append_chunk_text(merged, chunk_text)

        return self._post_process_text(merged)

    def _append_chunk_text(self, previous: str, current: str) -> str:
        previous = previous.strip()
        current = current.strip()

        if not previous:
            return current
        if not current:
            return previous

        previous_tokens = previous.split()
        current_tokens = current.split()

        max_overlap = min(80, len(previous_tokens), len(current_tokens))

        for size in range(max_overlap, 4, -1):
            if previous_tokens[-size:] == current_tokens[:size]:
                return " ".join(previous_tokens + current_tokens[size:])

        return f"{previous} {current}".strip()

    def _extract_audio_clip(self, input_path: Path, output_path: Path, start_sec: float, end_sec: float) -> None:
        duration = max(0.1, end_sec - start_sec)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start_sec:.3f}",
                    "-t",
                    f"{duration:.3f}",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            stderr_lines = [line.strip() for line in (error.stderr or "").splitlines() if line.strip()]
            detail = stderr_lines[-1] if stderr_lines else "ffmpeg clip extract failed"
            raise RuntimeError(f"AUDIO_CLIP_EXTRACT_FAILED: {detail}") from error

    def _preprocess_audio(self, input_path: Path) -> Path | None:
        output_path = input_path.with_name(f"{input_path.stem}_stt.wav")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
        ]

        if self.enable_loudnorm:
            command.extend(["-af", "loudnorm"])

        command.extend(
            [
                "-vn",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ]
        )

        started_at = time.perf_counter()
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            elapsed = time.perf_counter() - started_at
            print(
                "[STT] preprocess finished "
                f"path={output_path.name} "
                f"elapsed={elapsed:.1f}s "
                f"loudnorm={self.enable_loudnorm}"
            )
            return output_path
        except FileNotFoundError:
            if output_path.exists():
                output_path.unlink(missing_ok=True)
            return None
        except subprocess.CalledProcessError as error:
            if output_path.exists():
                output_path.unlink(missing_ok=True)

            stderr_lines = [line.strip() for line in (error.stderr or "").splitlines() if line.strip()]
            detail = stderr_lines[-1] if stderr_lines else "ffmpeg decode failed"
            raise RuntimeError(f"AUDIO_PREPROCESS_FAILED: {detail}") from error
