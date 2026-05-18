#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI
from tqdm import tqdm


BENCHMARK_CONFIGS = {
    "videoodyssey-v": {
        "annotation_suffix": "_V.json",
        "supports_audio": False,
    },
    "videoodyssey-av": {
        "annotation_suffix": "_AV.json",
        "supports_audio": True,
    },
}

def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Public benchmark runner for VideoOdyssey-style annotations."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(BENCHMARK_CONFIGS.keys()),
        help="Benchmark name. Controls annotation suffix validation and whether audio is allowed.",
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to annotation JSON. Must match the required suffix for --benchmark.",
    )
    parser.add_argument(
        "--videos_dir",
        required=True,
        help="Directory containing benchmark videos referenced by video_path.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Exact output JSON path. Existing outputs are resumed in place.",
    )
    parser.add_argument("--model", required=True, help="Model name passed to the OpenAI-compatible API.")
    parser.add_argument(
        "--input",
        required=True,
        choices=["v", "av", "video_subtitle"],
        help="Input modality: visual only, audio-visual, or video plus subtitles.",
    )
    parser.add_argument(
        "--subtitle_root",
        default="",
        help="Directory containing .srt files matched by video stem. Required for --input video_subtitle.",
    )
    parser.add_argument(
        "--use_certificate_window",
        type=parse_bool_arg,
        default=False,
        help="If true, only use the annotated time_reference window for each question.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=64,
        help="Number of sampled frames sent to the model.",
    )
    parser.add_argument(
        "--num_audio_samples",
        type=int,
        default=64,
        help="Number of audio segments used to build the montage track for av mode.",
    )
    parser.add_argument(
        "--audio_sample_len",
        type=int,
        default=10,
        help="Duration in seconds for each sampled audio segment in av mode.",
    )
    parser.add_argument(
        "--audio_bitrate",
        default="16k",
        help="Bitrate for ffmpeg audio extraction, e.g. 16k.",
    )
    parser.add_argument(
        "--jpeg_max_side",
        type=int,
        default=512,
        help="Resize frames so the longer side does not exceed this value before upload.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=70,
        help="JPEG quality used when encoding sampled frames.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    benchmark_config = BENCHMARK_CONFIGS[args.benchmark]
    annotation_name = Path(args.annotation).name
    required_suffix = benchmark_config["annotation_suffix"]

    if not annotation_name.endswith(required_suffix):
        raise ValueError(
            f"--benchmark {args.benchmark} requires --annotation to end with {required_suffix}, "
            f"but got {annotation_name}."
        )

    other_suffix = "_AV.json" if required_suffix == "_V.json" else "_V.json"
    if annotation_name.endswith(other_suffix):
        raise ValueError(
            f"--benchmark {args.benchmark} does not match annotation {annotation_name}. "
            f"Expected suffix {required_suffix}, not {other_suffix}."
        )

    if args.input == "av" and not benchmark_config["supports_audio"]:
        raise ValueError(f"--input av is not allowed for benchmark {args.benchmark}.")

    if args.input == "video_subtitle" and not args.subtitle_root:
        raise ValueError("--subtitle_root is required when --input video_subtitle is used.")

    if args.max_frames <= 0:
        raise ValueError("--max_frames must be positive.")
    if args.num_audio_samples <= 0:
        raise ValueError("--num_audio_samples must be positive.")
    if args.audio_sample_len <= 0:
        raise ValueError("--audio_sample_len must be positive.")
    if args.jpeg_max_side <= 0:
        raise ValueError("--jpeg_max_side must be positive.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be between 1 and 100.")

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Environment variable OPENAI_API_KEY is required.")


def init_client() -> OpenAI:
    client_kwargs: dict[str, Any] = {"api_key": os.environ["OPENAI_API_KEY"]}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    return OpenAI(**client_kwargs)


def _flatten_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            text = _flatten_message_content(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(content, dict):
        for key in ("text", "content", "value", "output_text", "response", "result"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, (list, dict)):
                nested = _flatten_message_content(value)
                if nested:
                    return nested
        return json.dumps(content, ensure_ascii=False)

    for attr in ("text", "content", "value"):
        value = getattr(content, attr, None)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, (list, dict)):
            nested = _flatten_message_content(value)
            if nested:
                return nested

    return str(content)


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            text = extract_response_text(dumped)
            if text:
                return text
        except Exception:
            pass

    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            text = _flatten_message_content(getattr(message, "content", message))
            if text:
                return text
        text = _flatten_message_content(getattr(first_choice, "text", None))
        if text:
            return text

    for attr in ("output_text", "text", "content", "response", "result"):
        value = getattr(response, attr, None)
        if value is not None:
            text = _flatten_message_content(value)
            if text:
                return text

    if isinstance(response, dict):
        return _flatten_message_content(response)

    return str(response)


def extract_answer_from_response_text(raw_response_text: str) -> str:
    res_text = str(raw_response_text).strip()
    if not res_text:
        return "Error"

    if "```json" in res_text:
        res_text = res_text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in res_text:
        res_text = res_text.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        res_json = json.loads(res_text)
        if isinstance(res_json, dict):
            raw_pred = res_json.get("predicted_answer", res_json.get("answer", ""))
            match = re.search(r"\b([A-D])\b", str(raw_pred).strip().upper())
            if match:
                return match.group(1)
    except json.JSONDecodeError:
        pass

    match = re.search(r'(?:predicted_)?answer["\']?\s*[:=]\s*["\']?([A-D])["\']?', res_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"\b([A-D])\b", res_text.upper())
    if match:
        return match.group(1)

    match = re.search(r"([A-D])", res_text.upper())
    if match:
        return match.group(1)

    return "Error"


def parse_time_to_seconds(time_text: str) -> float:
    parts = [int(part) for part in str(time_text).strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported time format: {time_text}")


def parse_time_reference(time_reference: str) -> tuple[float, float] | None:
    if not time_reference or "-" not in str(time_reference):
        return None
    start_text, end_text = [part.strip() for part in str(time_reference).split("-", 1)]
    start_sec = parse_time_to_seconds(start_text)
    end_sec = parse_time_to_seconds(end_text)
    if end_sec < start_sec:
        return None
    return start_sec, end_sec


def parse_srt_timestamp(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    seconds, millis = seconds.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000.0


def parse_srt_file(path: str) -> list[tuple[float, float, str]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    blocks = re.split(r"\n\s*\n", content)
    entries = []
    for block in blocks:
        lines = [line.strip("\ufeff") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line_index = 1 if re.match(r"^\d+$", lines[0]) else 0
        if time_line_index >= len(lines) or "-->" not in lines[time_line_index]:
            continue
        start_text, end_text = [part.strip() for part in lines[time_line_index].split("-->", 1)]
        text = " ".join(lines[time_line_index + 1 :]).strip()
        if not text:
            continue
        entries.append((parse_srt_timestamp(start_text), parse_srt_timestamp(end_text), text))
    return entries


def get_sampled_frame_timestamps(video_path: str, max_frames: int) -> list[float]:
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    if total_frames <= 0 or fps <= 0:
        return []

    interval = max(1, total_frames // max_frames)
    curr_frame = 0
    timestamps = []
    count = 0
    while count < max_frames and curr_frame < total_frames:
        timestamps.append(curr_frame / fps)
        curr_frame += interval
        count += 1
    return timestamps


def build_frame_timestamps(
    start_sec: float,
    end_sec: float,
    max_frames: int,
    per_second_if_short: bool,
) -> list[float]:
    duration = max(0.0, end_sec - start_sec)
    if duration <= 1e-6:
        return [start_sec]

    if per_second_if_short and duration < max_frames:
        frame_count = max(1, math.ceil(duration))
        timestamps = [start_sec + idx for idx in range(frame_count)]
    else:
        frame_count = max_frames
        step = duration / frame_count
        timestamps = [start_sec + idx * step for idx in range(frame_count)]

    upper_bound = max(start_sec, end_sec - 1e-3)
    return [min(ts, upper_bound) for ts in timestamps]


def get_subtitle_text_for_timestamps(entries: list[tuple[float, float, str]], timestamps: list[float]) -> str:
    aligned_subtitles = []
    for timestamp in timestamps:
        matched_text = ""
        for start, end, text in entries:
            if start <= timestamp <= end:
                matched_text = text
                break
        aligned_subtitles.append(json.dumps(matched_text, ensure_ascii=False))
    return "\n".join(aligned_subtitles)


def get_subtitle_text(video_path: str, subtitle_root: str, max_frames: int) -> str:
    video_stem = Path(video_path).stem
    subtitle_path = os.path.join(subtitle_root, f"{video_stem}.srt")
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found for {video_stem}: {subtitle_path}")

    entries = parse_srt_file(subtitle_path)
    if not entries:
        return ""

    timestamps = get_sampled_frame_timestamps(video_path, max_frames)
    return get_subtitle_text_for_timestamps(entries, timestamps)


def get_window_subtitle_text(video_path: str, subtitle_root: str, timestamps: list[float]) -> str:
    video_stem = Path(video_path).stem
    subtitle_path = os.path.join(subtitle_root, f"{video_stem}.srt")
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"Subtitle file not found for {video_stem}: {subtitle_path}")

    entries = parse_srt_file(subtitle_path)
    if not entries:
        return ""
    return get_subtitle_text_for_timestamps(entries, timestamps)


def resolve_effective_input_mode(
    video_path: str,
    requested_input_mode: str,
    subtitle_root: str,
    max_frames: int,
) -> tuple[str, str]:
    if requested_input_mode != "video_subtitle":
        return requested_input_mode, ""

    video_stem = Path(video_path).stem
    subtitle_path = os.path.join(subtitle_root, f"{video_stem}.srt")
    if not os.path.exists(subtitle_path):
        return "v", ""

    subtitle_text = get_subtitle_text(video_path, subtitle_root, max_frames)
    return "video_subtitle", subtitle_text


def encode_frame_to_base64(frame: Any, jpeg_max_side: int, jpeg_quality: int) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > jpeg_max_side:
        scale = jpeg_max_side / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def video_capture_fourcc(video: cv2.VideoCapture) -> str:
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((fourcc >> (8 * idx)) & 255) for idx in range(4))


def extract_frame_with_ffmpeg(
    video_path: str,
    timestamp: float,
    jpeg_max_side: int,
    jpeg_quality: int,
) -> str | None:
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        return None

    fd, frame_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    os.unlink(frame_path)
    try:
        cmd = [
            ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{max(timestamp, 0):.3f}",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "4",
            frame_path,
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=60)
        if result.returncode != 0 or not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
            return None
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        return encode_frame_to_base64(frame, jpeg_max_side, jpeg_quality)
    finally:
        if os.path.exists(frame_path):
            os.unlink(frame_path)


def extract_frames_with_ffmpeg(
    video_path: str,
    timestamps: list[float],
    jpeg_max_side: int,
    jpeg_quality: int,
) -> list[str]:
    frames = []
    for timestamp in timestamps:
        frame = extract_frame_with_ffmpeg(video_path, timestamp, jpeg_max_side, jpeg_quality)
        if frame is not None:
            frames.append(frame)
    return frames


def extract_frames(
    video_path: str,
    max_frames: int,
    jpeg_max_side: int,
    jpeg_quality: int,
) -> list[str]:
    video = cv2.VideoCapture(video_path)
    base64_frames: list[str] = []
    if not video.isOpened():
        return []

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        video.release()
        return []

    interval = max(1, total_frames // max_frames)
    if fps > 0 and video_capture_fourcc(video).upper() == "AV01":
        video.release()
        frame_count = min(max_frames, total_frames)
        timestamps = [(idx * interval) / fps for idx in range(frame_count)]
        return extract_frames_with_ffmpeg(video_path, timestamps, jpeg_max_side, jpeg_quality)

    curr_frame = 0
    count = 0
    while video.isOpened() and count < max_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            fallback_frame = (
                extract_frame_with_ffmpeg(video_path, curr_frame / fps, jpeg_max_side, jpeg_quality)
                if fps > 0
                else None
            )
            if fallback_frame is None:
                break
            base64_frames.append(fallback_frame)
        else:
            base64_frames.append(encode_frame_to_base64(frame, jpeg_max_side, jpeg_quality))
        count += 1
        curr_frame += interval

    video.release()
    return base64_frames


def extract_frames_by_timestamps(
    video_path: str,
    timestamps: list[float],
    jpeg_max_side: int,
    jpeg_quality: int,
) -> list[str]:
    video = cv2.VideoCapture(video_path)
    base64_frames: list[str] = []
    if not video.isOpened():
        return []

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        video.release()
        return []

    if video_capture_fourcc(video).upper() == "AV01":
        video.release()
        return extract_frames_with_ffmpeg(video_path, timestamps, jpeg_max_side, jpeg_quality)

    for timestamp in timestamps:
        frame_idx = min(max(int(round(timestamp * fps)), 0), max(total_frames - 1, 0))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = video.read()
        if not success:
            fallback_frame = extract_frame_with_ffmpeg(video_path, timestamp, jpeg_max_side, jpeg_quality)
            if fallback_frame is not None:
                base64_frames.append(fallback_frame)
            continue
        base64_frames.append(encode_frame_to_base64(frame, jpeg_max_side, jpeg_quality))

    video.release()
    return base64_frames


def extract_montage_audio(
    video_path: str,
    num_samples: int,
    sample_duration: int,
    audio_bitrate: str,
) -> str | None:
    cmd_dur = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        total_duration = float(subprocess.check_output(cmd_dur).decode().strip())
    except Exception:
        return None

    interval = total_duration / num_samples
    temp_dir = tempfile.mkdtemp(prefix="run_eval_av_")
    concat_list = os.path.join(temp_dir, "list.txt")
    final_mp3 = os.path.join(temp_dir, "montage.mp3")

    try:
        valid_clips = []
        for idx in range(num_samples):
            start = idx * interval
            clip_path = os.path.join(temp_dir, f"c_{idx}.mp3")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(sample_duration),
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-ab",
                audio_bitrate,
                "-loglevel",
                "error",
                clip_path,
            ]
            subprocess.run(cmd, check=False, timeout=30)
            if os.path.exists(clip_path):
                valid_clips.append(clip_path)

        if not valid_clips:
            return None

        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in valid_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", "-loglevel", "error", final_mp3],
            check=False,
            timeout=120,
        )
        if os.path.exists(final_mp3):
            with open(final_mp3, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return None


def extract_montage_audio_for_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_samples: int,
    sample_duration: int,
    audio_bitrate: str,
) -> str | None:
    total_duration = max(0.0, end_sec - start_sec)
    if total_duration <= 0:
        return None

    interval = total_duration / num_samples
    temp_dir = tempfile.mkdtemp(prefix="run_eval_av_window_")
    concat_list = os.path.join(temp_dir, "list.txt")
    final_mp3 = os.path.join(temp_dir, "montage.mp3")

    try:
        valid_clips = []
        for idx in range(num_samples):
            clip_start = start_sec + idx * interval
            remaining = max(0.0, end_sec - clip_start)
            clip_duration = min(sample_duration, remaining)
            if clip_duration <= 0:
                continue

            clip_path = os.path.join(temp_dir, f"c_{idx}.mp3")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(clip_start),
                "-t",
                str(clip_duration),
                "-i",
                video_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-ab",
                audio_bitrate,
                "-loglevel",
                "error",
                clip_path,
            ]
            subprocess.run(cmd, check=False, timeout=60)
            if os.path.exists(clip_path):
                valid_clips.append(clip_path)

        if not valid_clips:
            return None

        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in valid_clips:
                f.write(f"file '{os.path.abspath(clip)}'\n")

        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", "-loglevel", "error", final_mp3],
            check=False,
            timeout=120,
        )
        if os.path.exists(final_mp3):
            with open(final_mp3, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return None


def extract_audio_segment(video_path: str, start_sec: float, duration_sec: float, audio_bitrate: str) -> str | None:
    if duration_sec <= 0:
        return None

    temp_dir = tempfile.mkdtemp(prefix="run_eval_audio_seg_")
    clip_path = os.path.join(temp_dir, "segment.mp3")
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration_sec),
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-ab",
            audio_bitrate,
            "-loglevel",
            "error",
            clip_path,
        ]
        subprocess.run(cmd, check=False, timeout=120)
        if os.path.exists(clip_path):
            with open(clip_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return None


def get_video_duration(video_path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        return float(subprocess.check_output(cmd).decode().strip())
    except Exception:
        return None


def should_retry_question(question: dict[str, Any]) -> bool:
    return "predicted_answer" not in question


def merge_existing_results(
    base_data: list[dict[str, Any]],
    existing_data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_entries = {str(entry.get("video_id", "")): entry for entry in existing_data}
    for entry in base_data:
        existing_entry = existing_entries.get(str(entry.get("video_id", "")))
        if not existing_entry:
            continue
        existing_questions = {
            str(question.get("question_id", "")): question for question in existing_entry.get("questions", [])
        }
        for question in entry.get("questions", []):
            existing_question = existing_questions.get(str(question.get("question_id", "")))
            if not existing_question:
                continue
            if "predicted_answer" in existing_question:
                question["predicted_answer"] = existing_question["predicted_answer"]
    return base_data


def load_resume_data(annotation_path: str, output_path: str) -> list[dict[str, Any]]:
    with open(annotation_path, "r", encoding="utf-8") as f:
        base_data = copy.deepcopy(json.load(f))

    if not os.path.exists(output_path):
        return base_data

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
    except json.JSONDecodeError:
        return base_data

    return merge_existing_results(base_data, existing_data)


def save_json_atomically(data: list[dict[str, Any]], output_path: str) -> None:
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def build_prompt(question_data: dict[str, Any], input_mode: str, subtitle_text: str = "") -> str:
    q_text = str(question_data["question"])
    options_text = "\n".join(str(option) for option in question_data["options"])
    subtitle_text = str(subtitle_text)

    if input_mode == "av":
        return (
            "Based on the video and audio, please answer the question. "
            "You MUST choose the most appropriate answer from the given options (A, B, C, or D). "
            "Even if uncertain, make your best guess. Please directly output the option letter.\n\n"
            f"Question: {q_text}\nOptions:\n{options_text}"
        )
    elif input_mode == "video_subtitle":
        return (
            "The video's subtitles are listed below and aligned with the sampled video frames in order.\n"
            "Each line is one separate subtitle sentence enclosed in quotes.\n"
            "An empty quoted string means that sampled frame has no subtitle.\n\n"
            f"{subtitle_text}\n\n"
            "Based on the video and subtitles, please answer the question.\n"
            "You MUST choose the most appropriate answer from the given options (A, B, C, or D).\n"
            "Even if uncertain, make your best guess. Please directly output the option letter.\n\n"
            f"Question: {q_text}\nOptions:\n{options_text}"
        )
    else:
        return (
            "Based on the video only, please answer the question. "
            "You MUST choose the most appropriate answer from the given options (A, B, C, or D). "
            "Even if uncertain, make your best guess. Please directly output the option letter.\n\n"
            f"Question: {q_text}\nOptions:\n{options_text}"
        )


def ask_model(
    client: OpenAI,
    model_name: str,
    question_data: dict[str, Any],
    frames: list[str],
    input_mode: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    retry_delay: float,
    audio_b64: str | None = None,
    subtitle_text: str = "",
) -> str:
    user_prompt = build_prompt(question_data, input_mode, subtitle_text)

    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}",
                    "detail": "low",
                },
            }
        )
    if input_mode == "av" and audio_b64:
        content.append({"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}})

    for attempt in range(max_retries):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": temperature,
                "timeout": timeout,
            }
            response = client.chat.completions.create(**request_kwargs)
            raw = extract_response_text(response)
            return extract_answer_from_response_text(raw)
        except Exception as exc:
            error_msg = str(exc)
            tqdm.write(f"[RETRY {attempt + 1}/{max_retries}] {type(exc).__name__}: {error_msg[:300]}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    return "API_Error"


def build_question_evidence(
    video_path: str,
    question_data: dict[str, Any],
    requested_input_mode: str,
    subtitle_root: str,
    use_certificate_window: bool,
    max_frames: int,
    num_audio_samples: int,
    audio_sample_len: int,
    audio_bitrate: str,
    jpeg_max_side: int,
    jpeg_quality: int,
) -> tuple[list[str], str | None, str, str]:
    effective_input_mode = requested_input_mode
    subtitle_text = ""
    audio_b64 = None

    if not use_certificate_window:
        effective_input_mode, subtitle_text = resolve_effective_input_mode(
            video_path,
            requested_input_mode,
            subtitle_root,
            max_frames,
        )
        frames = extract_frames(video_path, max_frames, jpeg_max_side, jpeg_quality)
        if effective_input_mode == "av":
            audio_b64 = extract_montage_audio(video_path, num_audio_samples, audio_sample_len, audio_bitrate)
        return frames, audio_b64, effective_input_mode, subtitle_text

    window = parse_time_reference(question_data.get("time_reference", ""))
    if not window:
        effective_input_mode, subtitle_text = resolve_effective_input_mode(
            video_path,
            requested_input_mode,
            subtitle_root,
            max_frames,
        )
        frames = extract_frames(video_path, max_frames, jpeg_max_side, jpeg_quality)
        if effective_input_mode == "av":
            audio_b64 = extract_montage_audio(video_path, num_audio_samples, audio_sample_len, audio_bitrate)
        return frames, audio_b64, effective_input_mode, subtitle_text

    start_sec, end_sec = window
    timestamps = build_frame_timestamps(start_sec, end_sec, max_frames, per_second_if_short=True)
    frames = extract_frames_by_timestamps(video_path, timestamps, jpeg_max_side, jpeg_quality)
    duration = end_sec - start_sec

    if requested_input_mode == "av":
        if duration <= 1e-6:
            video_duration = get_video_duration(video_path)
            clip_start = max(0.0, start_sec - 0.5)
            if video_duration is None:
                clip_end = start_sec + 0.5
            else:
                clip_end = min(video_duration, start_sec + 0.5)
            if clip_end <= clip_start:
                clip_end = clip_start + 1.0
                if video_duration is not None:
                    clip_end = min(video_duration, clip_end)
            single_point_duration = max(0.0, clip_end - clip_start)
            audio_b64 = extract_audio_segment(video_path, clip_start, single_point_duration, audio_bitrate)
        elif duration < max_frames:
            audio_b64 = extract_audio_segment(video_path, start_sec, duration, audio_bitrate)
        elif duration <= num_audio_samples * audio_sample_len:
            audio_b64 = extract_audio_segment(video_path, start_sec, duration, audio_bitrate)
        else:
            audio_b64 = extract_montage_audio_for_window(
                video_path,
                start_sec,
                end_sec,
                num_audio_samples,
                audio_sample_len,
                audio_bitrate,
            )
    elif requested_input_mode == "video_subtitle":
        video_stem = Path(video_path).stem
        subtitle_path = os.path.join(subtitle_root, f"{video_stem}.srt")
        if os.path.exists(subtitle_path):
            subtitle_text = get_window_subtitle_text(video_path, subtitle_root, timestamps)
            effective_input_mode = "video_subtitle"
        else:
            effective_input_mode = "v"

    return frames, audio_b64, effective_input_mode, subtitle_text


def prepare_full_video_cache(
    entry: dict[str, Any],
    video_path: str,
    requested_input_mode: str,
    subtitle_root: str,
    max_frames: int,
    num_audio_samples: int,
    audio_sample_len: int,
    audio_bitrate: str,
    jpeg_max_side: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    effective_input_mode, subtitle_text = resolve_effective_input_mode(
        video_path,
        requested_input_mode,
        subtitle_root,
        max_frames,
    )
    cache = {
        "frames": extract_frames(video_path, max_frames, jpeg_max_side, jpeg_quality),
        "audio_b64": None,
        "effective_input_mode": effective_input_mode,
        "subtitle_text": subtitle_text,
    }
    if effective_input_mode == "av":
        cache["audio_b64"] = extract_montage_audio(
            video_path,
            num_audio_samples,
            audio_sample_len,
            audio_bitrate,
        )
    return cache


def run_eval(args: argparse.Namespace) -> None:
    client = init_client()
    data = load_resume_data(args.annotation, args.output_path)

    for entry in tqdm(data, desc="Videos"):
        questions = entry.get("questions", [])
        pending_questions = [q for q in questions if should_retry_question(q)]
        if not pending_questions:
            continue

        rel_video_path = entry.get("video_path", "")
        full_video_path = os.path.join(args.videos_dir, rel_video_path)
        if not os.path.exists(full_video_path):
            tqdm.write(f"[VIDEO MISSING] {full_video_path}")
            for question in pending_questions:
                question["predicted_answer"] = "Video_Not_Found"
            save_json_atomically(data, args.output_path)
            continue

        full_video_cache = None
        if not args.use_certificate_window:
            try:
                full_video_cache = prepare_full_video_cache(
                    entry,
                    full_video_path,
                    args.input,
                    args.subtitle_root,
                    args.max_frames,
                    args.num_audio_samples,
                    args.audio_sample_len,
                    args.audio_bitrate,
                    args.jpeg_max_side,
                    args.jpeg_quality,
                )
            except Exception as exc:
                tqdm.write(f"[EVIDENCE FAILED] {rel_video_path}: {exc}")
                continue

        for question in tqdm(questions, desc="Questions", leave=False):
            if not should_retry_question(question):
                continue

            try:
                if full_video_cache is not None:
                    frames = full_video_cache["frames"]
                    audio_b64 = full_video_cache["audio_b64"]
                    effective_input_mode = full_video_cache["effective_input_mode"]
                    subtitle_text = full_video_cache["subtitle_text"]
                else:
                    frames, audio_b64, effective_input_mode, subtitle_text = build_question_evidence(
                        full_video_path,
                        question,
                        args.input,
                        args.subtitle_root,
                        args.use_certificate_window,
                        args.max_frames,
                        args.num_audio_samples,
                        args.audio_sample_len,
                        args.audio_bitrate,
                        args.jpeg_max_side,
                        args.jpeg_quality,
                    )
            except Exception as exc:
                tqdm.write(
                    f"[EVIDENCE FAILED] Video: {rel_video_path} | QID: {question.get('question_id')} | {exc}"
                )
                continue

            if args.input == "video_subtitle" and effective_input_mode == "v":
                tqdm.write(
                    f"[SUBTITLE MISSING] Video: {rel_video_path} | QID: {question.get('question_id')} | fallback to v"
                )

            if effective_input_mode == "av":
                if not frames or not audio_b64:
                    tqdm.write(f"[EVIDENCE EMPTY] Video: {rel_video_path} | QID: {question.get('question_id')}")
                    question["predicted_answer"] = "API_Error"
                    save_json_atomically(data, args.output_path)
                    continue
            elif not frames:
                tqdm.write(f"[FRAME EMPTY] Video: {rel_video_path} | QID: {question.get('question_id')}")
                question["predicted_answer"] = "API_Error"
                save_json_atomically(data, args.output_path)
                continue

            predicted_answer = ask_model(
                client=client,
                model_name=args.model,
                question_data=question,
                frames=frames,
                input_mode=effective_input_mode,
                audio_b64=audio_b64,
                subtitle_text=subtitle_text,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
            )
            question["predicted_answer"] = predicted_answer
            save_json_atomically(data, args.output_path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    run_eval(args)


if __name__ == "__main__":
    main()
