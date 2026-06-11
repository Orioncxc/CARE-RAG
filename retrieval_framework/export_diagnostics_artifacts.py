from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def as_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return default


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def bar_chart_svg(
    path: Path,
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    counts: Optional[Sequence[int]] = None,
    width: int = 760,
    height: int = 460,
    color: str = "#3b6ea8",
) -> None:
    margin_left = 86
    margin_right = 34
    margin_top = 74
    margin_bottom = 86
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = 1.0
    bar_gap = 36
    bar_width = max(42, int((plot_width - bar_gap * (len(values) + 1)) / max(1, len(values))))

    def x_pos(index: int) -> float:
        total_bar = bar_width * len(values)
        total_gap = bar_gap * (len(values) - 1)
        start = margin_left + (plot_width - total_bar - total_gap) / 2
        return start + index * (bar_width + bar_gap)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1 - min(max(value, 0.0), max_value) / max_value)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{svg_text(title)}</text>',
    ]

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = y_pos(tick)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#d7dde5" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#334155">{int(tick * 100)}%</text>'
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#334155" stroke-width="1.2"/>'
    )
    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#334155" stroke-width="1.2"/>'
    )

    for index, (label, value) in enumerate(zip(labels, values)):
        x = x_pos(index)
        y = y_pos(value)
        h = margin_top + plot_height - y
        lines.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{h:.1f}" fill="{color}" rx="3"/>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#0f172a">{pct(value)}</text>'
        )
        count_label = f"n={counts[index]}" if counts else ""
        label_y = margin_top + plot_height + 30
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{label_y:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#0f172a">{svg_text(label)}</text>'
        )
        if count_label:
            lines.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{label_y + 22:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#64748b">{svg_text(count_label)}</text>'
            )

    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 24 {margin_top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">Attack success rate</text>'
    )
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def horizontal_bar_svg(
    path: Path,
    title: str,
    rows: Sequence[Tuple[str, float, int]],
    width: int = 860,
    height: Optional[int] = None,
) -> None:
    height = height or max(320, 118 + len(rows) * 58)
    margin_left = 260
    margin_right = 70
    margin_top = 72
    margin_bottom = 40
    plot_width = width - margin_left - margin_right
    bar_height = 26
    row_gap = 30
    max_value = max([value for _, value, _ in rows] or [1.0, 1.0])
    max_value = max(max_value, 0.01)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{svg_text(title)}</text>',
    ]
    for index, (label, value, count) in enumerate(rows):
        y = margin_top + index * (bar_height + row_gap)
        bar_width = plot_width * value / max_value
        lines.append(
            f'<text x="{margin_left - 14}" y="{y + 18}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{svg_text(label)}</text>'
        )
        lines.append(
            f'<rect x="{margin_left}" y="{y}" width="{bar_width:.1f}" height="{bar_height}" fill="#7a4f9f" rx="3"/>'
        )
        lines.append(
            f'<text x="{margin_left + bar_width + 10:.1f}" y="{y + 18}" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{pct(value)} ({count})</text>'
        )
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def heatmap_svg(
    path: Path,
    title: str,
    pattern_rows: Sequence[Dict[str, str]],
    width: int = 820,
    height: int = 620,
) -> None:
    cells: Dict[Tuple[int, int], Dict[str, float]] = {}
    clean_values = set()
    poison_values = set()
    for row in pattern_rows:
        clean = as_int(row, "clean_gold_support_count")
        poison = as_int(row, "poison_support_count")
        clean_values.add(clean)
        poison_values.add(poison)
        item = cells.setdefault((clean, poison), {"asr_count": 0.0, "num_queries": 0.0})
        item["asr_count"] += as_float(row, "asr_count")
        item["num_queries"] += as_float(row, "num_queries")

    clean_axis = sorted(clean_values)
    poison_axis = sorted(poison_values)
    margin_left = 112
    margin_right = 36
    margin_top = 84
    margin_bottom = 92
    cell_w = (width - margin_left - margin_right) / max(1, len(poison_axis))
    cell_h = (height - margin_top - margin_bottom) / max(1, len(clean_axis))

    def color(value: float) -> str:
        value = min(max(value, 0.0), 1.0)
        low = (232, 242, 249)
        high = (184, 66, 66)
        rgb = tuple(int(low[i] + (high[i] - low[i]) * value) for i in range(3))
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{svg_text(title)}</text>',
    ]
    for y_index, clean in enumerate(reversed(clean_axis)):
        for x_index, poison in enumerate(poison_axis):
            x = margin_left + x_index * cell_w
            y = margin_top + y_index * cell_h
            item = cells.get((clean, poison), {"asr_count": 0.0, "num_queries": 0.0})
            n = int(item["num_queries"])
            rate = item["asr_count"] / item["num_queries"] if item["num_queries"] else 0.0
            fill = color(rate) if n else "#f8fafc"
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="white" stroke-width="2"/>'
            )
            if n:
                lines.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 - 4:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#0f172a">{pct(rate)}</text>'
                )
                lines.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 15:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#334155">n={n}</text>'
                )
        y_label = margin_top + y_index * cell_h + cell_h / 2 + 5
        lines.append(
            f'<text x="{margin_left - 16}" y="{y_label:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{clean}</text>'
        )

    for x_index, poison in enumerate(poison_axis):
        x_label = margin_left + x_index * cell_w + cell_w / 2
        lines.append(
            f'<text x="{x_label:.1f}" y="{margin_top + len(clean_axis) * cell_h + 28:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{poison}</text>'
        )

    lines.append(
        f'<text x="{margin_left + len(poison_axis) * cell_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">Poison support count</text>'
    )
    lines.append(
        f'<text x="26" y="{margin_top + len(clean_axis) * cell_h / 2:.1f}" transform="rotate(-90 26 {margin_top + len(clean_axis) * cell_h / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">Clean gold support count</text>'
    )
    lines.append("</svg>")
    write_text(path, "\n".join(lines))


def latex_table(path: Path, caption: str, label: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    colspec = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    write_text(path, "\n".join(lines))


def make_rank_table_rows(rows: Sequence[Dict[str, str]], key: str, false_label: str, true_label: str) -> List[List[Any]]:
    output = []
    for row in sorted(rows, key=lambda item: as_int(item, key)):
        label = true_label if as_int(row, key) else false_label
        output.append(
            [
                label,
                as_int(row, "num_queries"),
                as_int(row, "asr_count"),
                pct(as_float(row, "asr_rate")),
                f"{as_float(row, 'mean_target_adv_count_topk'):.2f}",
                f"{as_float(row, 'mean_clean_gold_support_count'):.2f}",
                f"{as_float(row, 'mean_poison_support_count'):.2f}",
            ]
        )
    return output


def write_markdown_summary(
    path: Path,
    table1: Sequence[Dict[str, str]],
    table2: Sequence[Dict[str, str]],
    priority: Sequence[Dict[str, str]],
) -> None:
    def rate(rows: Sequence[Dict[str, str]], key: str, value: int) -> str:
        for row in rows:
            if as_int(row, key) == value:
                return pct(as_float(row, "asr_rate"))
        return "n/a"

    lines = [
        "# Paper-ready diagnostics artifacts",
        "",
        "Generated files in this directory:",
        "",
        "- `fig_asr_rank1.svg`: ASR split by whether a target poison document is rank 1.",
        "- `fig_asr_top3.svg`: ASR split by whether a target poison document appears in top 3.",
        "- `fig_support_heatmap.svg`: ASR by clean gold support count and poison support count.",
        "- `fig_error_buckets.svg`: ASR error bucket distribution.",
        "- `table_*.tex`: LaTeX tables for paper drafts.",
        "",
        "Suggested paper text:",
        "",
        (
            f"In our diagnostic run, ASR increases from {rate(table1, 'poison_rank1', 0)} "
            f"to {rate(table1, 'poison_rank1', 1)} when a target poisoned document is ranked first. "
            f"When a target poisoned document appears in the top three retrieved passages, ASR is "
            f"{rate(table2, 'poison_top3', 1)}, compared with {rate(table2, 'poison_top3', 0)} otherwise. "
            "This indicates that attack success is primarily driven by top-ranked evidence dominance rather "
            "than by near-duplicate poisoned clusters."
        ),
        "",
        "Main priority buckets:",
        "",
    ]
    for row in priority:
        lines.append(
            f"- {row.get('bucket')}: {as_int(row, 'asr_error_count')} cases ({pct(as_float(row, 'share_of_asr_errors'))})."
        )
    lines.append("")
    write_text(path, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ASR diagnostics CSV files as paper-ready SVG figures and LaTeX tables."
    )
    parser.add_argument(
        "--diagnostics_prefix",
        required=True,
        help="Prefix of diagnostics files, e.g. results/diagnostics/nq-evidence_hardened-asr",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to results/paper_artifacts/<prefix-name>.",
    )
    parser.add_argument("--top_patterns", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = Path(args.diagnostics_prefix)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = prefix.parent.parent / "paper_artifacts" / prefix.name
    output_dir.mkdir(parents=True, exist_ok=True)

    table1 = read_csv(prefix.with_suffix(".table1_rank1.csv"))
    table2 = read_csv(prefix.with_suffix(".table2_top3.csv"))
    patterns = read_csv(prefix.with_suffix(".table3_support_patterns.csv"))
    priority = read_csv(prefix.with_suffix(".table4_error_priority_buckets.csv"))
    multilabel = read_csv(prefix.with_suffix(".table4_error_multilabel.csv"))

    rank1_labels = ["No rank-1 poison", "Rank-1 poison"]
    rank1_rows = sorted(table1, key=lambda row: as_int(row, "poison_rank1"))
    bar_chart_svg(
        output_dir / "fig_asr_rank1.svg",
        "ASR by poison at rank 1",
        rank1_labels,
        [as_float(row, "asr_rate") for row in rank1_rows],
        [as_int(row, "num_queries") for row in rank1_rows],
        color="#2f6f9f",
    )

    top3_labels = ["No top-3 poison", "Top-3 poison"]
    top3_rows = sorted(table2, key=lambda row: as_int(row, "poison_top3"))
    bar_chart_svg(
        output_dir / "fig_asr_top3.svg",
        "ASR by poison in top 3",
        top3_labels,
        [as_float(row, "asr_rate") for row in top3_rows],
        [as_int(row, "num_queries") for row in top3_rows],
        color="#b05d3b",
    )

    heatmap_svg(
        output_dir / "fig_support_heatmap.svg",
        "ASR by clean and poison support",
        patterns,
    )

    bucket_rows = [
        (
            row.get("label") or row.get("bucket", ""),
            as_float(row, "share_of_asr_errors"),
            as_int(row, "asr_error_count"),
        )
        for row in multilabel
    ]
    horizontal_bar_svg(
        output_dir / "fig_error_buckets.svg",
        "ASR error diagnostics",
        bucket_rows,
    )

    latex_table(
        output_dir / "table_asr_rank1.tex",
        "Attack success rate conditioned on target poison evidence at rank 1.",
        "tab:asr-rank1",
        ["Condition", "Queries", "ASR count", "ASR", "Adv/top-k", "Clean support", "Poison support"],
        make_rank_table_rows(table1, "poison_rank1", "No rank-1 poison", "Rank-1 poison"),
    )
    latex_table(
        output_dir / "table_asr_top3.tex",
        "Attack success rate conditioned on target poison evidence in the top three retrieved passages.",
        "tab:asr-top3",
        ["Condition", "Queries", "ASR count", "ASR", "Adv/top-k", "Clean support", "Poison support"],
        make_rank_table_rows(table2, "poison_top3", "No top-3 poison", "Top-3 poison"),
    )

    pattern_rows = []
    for row in patterns[: args.top_patterns]:
        pattern_rows.append(
            [
                as_int(row, "clean_gold_support_count"),
                as_int(row, "poison_support_count"),
                as_int(row, "poison_rank1"),
                as_int(row, "poison_top3"),
                as_int(row, "num_queries"),
                as_int(row, "asr_count"),
                pct(as_float(row, "asr_rate")),
            ]
        )
    latex_table(
        output_dir / "table_support_patterns.tex",
        "Most frequent clean-support and poison-support patterns.",
        "tab:support-patterns",
        ["Clean", "Poison", "Rank1", "Top3", "Queries", "ASR count", "ASR"],
        pattern_rows,
    )

    bucket_table_rows = [
        [row.get("bucket", ""), as_int(row, "asr_error_count"), pct(as_float(row, "share_of_asr_errors"))]
        for row in priority
    ]
    latex_table(
        output_dir / "table_error_buckets.tex",
        "Priority buckets for successful attack cases.",
        "tab:error-buckets",
        ["Bucket", "Errors", "Share"],
        bucket_table_rows,
    )

    write_markdown_summary(output_dir / "paper_diagnostics_summary.md", table1, table2, priority)

    print(f"Wrote paper artifacts to: {output_dir}")
    for name in [
        "fig_asr_rank1.svg",
        "fig_asr_top3.svg",
        "fig_support_heatmap.svg",
        "fig_error_buckets.svg",
        "table_asr_rank1.tex",
        "table_asr_top3.tex",
        "table_support_patterns.tex",
        "table_error_buckets.tex",
        "paper_diagnostics_summary.md",
    ]:
        print(f"- {output_dir / name}")


if __name__ == "__main__":
    main()
