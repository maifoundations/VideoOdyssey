import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


DEFAULT_RESULT_PATH = "path/to/result.json"
DEFAULT_BENCHMARK_PATH = "path/to/annotation.json"

VIDEO_DURATION_ORDER = [
    "[60, 90) min",
    "[90, 120) min",
    "[120, 180) min",
    "[180, 270) min",
    "Other/Short",
]

CONTINUOUS_CERTIFICATE_LENGTH_ORDER = [
    "[0, 0.5) min",
    "[0.5, 3) min",
    "[3, 15) min",
    "[15, 60) min",
    "[60, inf) min",
    "Unknown",
]


def calculate_accuracy(correct, total):
    return (correct / total * 100) if total > 0 else 0.0


def parse_time_to_seconds(time_str):
    """Parse a time string into seconds. Supported formats: 45, 01:30, 01:02:30."""
    try:
        parts = [int(part) for part in str(time_str).strip().split(":")]
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:
        return 0
    return 0


def get_span_from_reference(time_reference):
    """Return the duration in seconds from a time_reference such as 00:00-00:45."""
    if not time_reference or "-" not in str(time_reference):
        return 0
    try:
        start_str, end_str = str(time_reference).split("-", 1)
        start_sec = parse_time_to_seconds(start_str)
        end_sec = parse_time_to_seconds(end_str)
        return max(0, end_sec - start_sec)
    except Exception:
        return 0


def get_video_duration_bin(duration_minutes):
    minutes = duration_minutes
    if 60 <= minutes < 90:
        return "[60, 90) min"
    if 90 <= minutes < 120:
        return "[90, 120) min"
    if 120 <= minutes < 180:
        return "[120, 180) min"
    if 180 <= minutes < 270:
        return "[180, 270) min"
    return "Other/Short"


def get_span_bin(span_sec):
    minutes = span_sec / 60
    if minutes < 0.5:
        return "[0, 0.5) min"
    if 0.5 <= minutes < 3:
        return "[0.5, 3) min"
    if 3 <= minutes < 15:
        return "[3, 15) min"
    if 15 <= minutes < 60:
        return "[15, 60) min"
    if minutes >= 60:
        return "[60, inf) min"
    return "Unknown"


def sort_labels(labels, preferred_order=None):
    if preferred_order:
        order_index = {label: index for index, label in enumerate(preferred_order)}
        return sorted(labels, key=lambda label: (order_index.get(label, len(order_index)), label))
    return sorted(labels)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_save_path(save_path, result_path):
    default_name = f"{Path(result_path).stem}_detailed_stats.txt"
    target_dir = save_path or str(Path(result_path).parent)
    return str(Path(target_dir) / default_name)


def normalize_question_types(question_types):
    if not question_types:
        return []
    if not isinstance(question_types, list):
        question_types = [question_types]
    return [str(item).strip() for item in question_types if str(item).strip()]


def parse_question_type_filter(raw_value):
    if not raw_value:
        return set()
    return {
        item.lower()
        for item in (part.strip() for part in str(raw_value).split(","))
        if item
    }


def should_skip_question(question_types, include_types=None, exclude_types=None):
    normalized = {item.lower() for item in normalize_question_types(question_types)}
    if include_types and not (normalized & include_types):
        return True
    if exclude_types and (normalized & exclude_types):
        return True
    return False


def get_prediction(question):
    for field in ("predicted_answer", "pred_answer"):
        value = str(question.get(field, "")).strip().upper()
        if value:
            return value
    answer_and_evl = question.get("answer_and_evl")
    if isinstance(answer_and_evl, dict):
        value = str(answer_and_evl.get("final_answer", "")).strip().upper()
        if value:
            return value
    return ""


def parse_correctness_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def get_question_correctness(question, ground_truth):
    for field in ("is_true", "is_correct"):
        parsed = parse_correctness_flag(question.get(field))
        if parsed is not None:
            return parsed, False

    prediction = get_prediction(question)
    if prediction not in {"A", "B", "C", "D"}:
        return None, True
    return prediction == ground_truth, False


def iter_result_questions(result_data):
    if not isinstance(result_data, list):
        return

    for item in result_data:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("questions"), list):
            for question in item.get("questions", []):
                if isinstance(question, dict):
                    yield question
            continue
        if "question_id" in item:
            yield item


def build_benchmark_index(benchmark_data):
    benchmark_video_meta = {}
    benchmark_questions = {}
    has_audio_type = False

    for video in benchmark_data:
        video_id = video.get("video_id")
        benchmark_video_meta[video_id] = {
            "video_category": video.get("video_category", "Unknown"),
            "duration_minutes": video.get("duration_minutes", 0),
        }
        for question in video.get("questions", []):
            question_info = {
                "video_id": video_id,
                "answer": str(question.get("answer", "")).strip().upper(),
                "question_type": question.get("question_type", ["Unknown"]),
                "time_reference": question.get("time_reference", ""),
            }
            if "audio_type" in question:
                has_audio_type = True
                question_info["audio_type"] = question.get("audio_type", ["Unknown"])
            benchmark_questions[question.get("question_id")] = question_info

    return benchmark_video_meta, benchmark_questions, has_audio_type


def initialize_stats(has_audio_type):
    stats = {
        "Question Type": defaultdict(lambda: [0, 0]),
        "Video Duration": defaultdict(lambda: [0, 0]),
        "Continuous Certificate Length": defaultdict(lambda: [0, 0]),
        "Video Category": defaultdict(lambda: [0, 0]),
    }
    if has_audio_type:
        stats["Audio Type"] = defaultdict(lambda: [0, 0])
    return stats


def update_dimension_stats(stats, dimension, labels, is_correct):
    if not isinstance(labels, list):
        labels = [labels]
    labels = labels or ["Unknown"]
    for label in labels:
        label_str = str(label)
        stats[dimension][label_str][1] += 1
        if is_correct:
            stats[dimension][label_str][0] += 1


def analyze_results(
    result_path,
    benchmark_path,
    save_path=None,
    include_question_types=None,
    exclude_question_types=None,
):
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result file not found: {result_path}")
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    result_data = load_json(result_path)
    benchmark_data = load_json(benchmark_path)
    benchmark_video_meta, benchmark_questions, has_audio_type = build_benchmark_index(benchmark_data)

    stats = initialize_stats(has_audio_type)
    total_correct = 0
    total_questions = 0

    skipped_not_in_benchmark = 0
    skipped_excluded_type = 0
    skipped_invalid_prediction = 0
    seen_question_ids = set()

    for question in iter_result_questions(result_data):
        question_id = question.get("question_id")
        if not question_id or question_id in seen_question_ids:
            continue
        seen_question_ids.add(question_id)

        benchmark_question = benchmark_questions.get(question_id)
        if benchmark_question is None:
            skipped_not_in_benchmark += 1
            continue

        question_types = benchmark_question.get("question_type", ["Unknown"])
        if should_skip_question(
            question_types,
            include_types=include_question_types,
            exclude_types=exclude_question_types,
        ):
            skipped_excluded_type += 1
            continue

        ground_truth = benchmark_question.get("answer", "")
        is_correct, should_skip_invalid_prediction = get_question_correctness(question, ground_truth)
        if should_skip_invalid_prediction:
            skipped_invalid_prediction += 1
            continue

        total_questions += 1
        if is_correct:
            total_correct += 1

        video_meta = benchmark_video_meta.get(
            benchmark_question.get("video_id"),
            {"video_category": "Unknown", "duration_minutes": 0},
        )
        video_category = video_meta.get("video_category", "Unknown")
        duration_bin = get_video_duration_bin(video_meta.get("duration_minutes", 0))
        span_sec = get_span_from_reference(benchmark_question.get("time_reference", ""))
        span_bin = get_span_bin(span_sec)

        update_dimension_stats(stats, "Question Type", question_types, is_correct)
        if has_audio_type:
            update_dimension_stats(
                stats, "Audio Type", benchmark_question.get("audio_type", ["Unknown"]), is_correct
            )
        update_dimension_stats(stats, "Video Duration", [duration_bin], is_correct)
        update_dimension_stats(stats, "Continuous Certificate Length", [span_bin], is_correct)
        update_dimension_stats(stats, "Video Category", [video_category], is_correct)

    report = []
    report.append("=" * 70)
    report.append("OmniLong Benchmark Detailed Accuracy Report")
    report.append(f"Result file: {result_path}")
    report.append(f"Benchmark file: {benchmark_path}")
    report.append(
        "Question type filter: "
        f"include={', '.join(sorted(include_question_types)) if include_question_types else 'ALL'}; "
        f"exclude={', '.join(sorted(exclude_question_types)) if exclude_question_types else 'NONE'}"
    )
    report.append(
        f"Evaluated questions: {total_questions} | Overall accuracy: {calculate_accuracy(total_correct, total_questions):.2f}%"
    )
    report.append(
        "Skipped: "
        f"not_in_benchmark={skipped_not_in_benchmark}, "
        f"excluded_question_type={skipped_excluded_type}, "
        f"invalid_prediction={skipped_invalid_prediction}"
    )
    report.append("=" * 70)

    order = ["Video Category", "Question Type"]
    if has_audio_type:
        order.append("Audio Type")
    order.extend(["Video Duration", "Continuous Certificate Length"])
    sort_orders = {
        "Video Duration": VIDEO_DURATION_ORDER,
        "Continuous Certificate Length": CONTINUOUS_CERTIFICATE_LENGTH_ORDER,
    }
    for dimension in order:
        details = stats[dimension]
        report.append(f"\n[Dimension: {dimension}]")
        report.append(f"{'Subcategory':<36} | {'Correct/Total':<12} | {'Accuracy':<8}")
        report.append("-" * 65)
        for label in sort_labels(details.keys(), sort_orders.get(dimension)):
            correct, total = details[label]
            accuracy = calculate_accuracy(correct, total)
            report.append(f"{label:<36} | {f'{correct}/{total}':<12} | {accuracy:>6.2f}%")

    final_text = "\n".join(report)
    print(final_text)

    if save_path:
        save_parent = os.path.dirname(save_path)
        if save_parent:
            os.makedirs(save_parent, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as file:
            file.write(final_text)
        print(f"\n[Done] Report saved to: {save_path}")

    return {
        "total_questions": total_questions,
        "total_correct": total_correct,
        "accuracy": calculate_accuracy(total_correct, total_questions),
        "skipped_not_in_benchmark": skipped_not_in_benchmark,
        "skipped_excluded_type": skipped_excluded_type,
        "skipped_invalid_prediction": skipped_invalid_prediction,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy on the OmniLong benchmark and report detailed statistics by dimension."
    )
    parser.add_argument(
        "--result-path",
        default=DEFAULT_RESULT_PATH,
        help=f"Path to the result file. Default: {DEFAULT_RESULT_PATH}",
    )
    parser.add_argument(
        "--benchmark-path",
        default=DEFAULT_BENCHMARK_PATH,
        help=f"Path to the benchmark annotation file. Default: {DEFAULT_BENCHMARK_PATH}",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Output directory for the report. The file name is derived from the result file name.",
    )
    parser.add_argument(
        "--include-question-types",
        default=None,
        help="Only evaluate these question types, separated by commas. Default: evaluate all.",
    )
    parser.add_argument(
        "--exclude-question-types",
        default=None,
        help="Exclude these question types, separated by commas. Default: exclude none.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    save_path = resolve_save_path(args.save_path, args.result_path)
    analyze_results(
        args.result_path,
        args.benchmark_path,
        save_path=save_path,
        include_question_types=parse_question_type_filter(args.include_question_types),
        exclude_question_types=parse_question_type_filter(args.exclude_question_types),
    )
