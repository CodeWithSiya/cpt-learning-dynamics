"""
Plot downstream learning dynamics as panel grids spanning every model, language and task.
"""

import argparse
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np

import style

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = ["xho", "zul"]

TASK_DISPLAY_NAMES = {
    "ner": "NER",
    "pos": "POS",
    "ntc": "NTC",
}

VALUE_LABEL = "Macro F1"

CLASS_DISPLAY_NAMES = {
    "Ezemidlalo": "Sports",
    "Ezezimoto": "Motoring",
    "Ezokungcebeleka": "Entertainment",
    "Imibono": "Opinion",
    "Intandokazi": "Lifestyle",
}

POS_GROUPS = {
    "Nouns": ["NOUN", "PROPN"],
    "Verbs": ["VERB", "AUX"],
    "Modifiers": ["ADJ", "ADV"],
    "Function": ["ADP", "DET", "PRON", "CCONJ", "SCONJ", "PART"],
    "Punctuation": ["PUNCT", "SYM"],
    "Other": ["INTJ", "NUM", "X"],
}
STEP_LABEL = "CPT Step"

def parse_args() -> Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot downstream learning dynamics as panel grids."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Root results directory containing the per-model result folders."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(style.MODEL_DISPLAY_NAMES),
        choices=list(style.MODEL_DISPLAY_NAMES),
        help="Models to plot, in column order. Defaults to every model."
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=SUPPORTED_LANGUAGES,
        choices=SUPPORTED_LANGUAGES,
        help="Languages to plot, in row order. Defaults to every language."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=list(TASK_DISPLAY_NAMES),
        choices=list(TASK_DISPLAY_NAMES),
        help="Tasks to plot, in column order. Defaults to every task."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the figures to."
    )
    return parser.parse_args()

def task_filename(task: str, language: str) -> str:
    """
    Resolve a task to the aggregated results filename it was written under.

    News topic classification is language specific, so its results are stored
    per language rather than under a single shared task name.

    :param task: Task name as used in this script, e.g. "ntc".
    :param language: Language code, e.g. "zul".
    :return: Stem of the aggregated results file for that task and language.
    """
    return f"ntc_{language}" if task == "ntc" else task

def load_aggregated(results_dir: Path, model: str, language: str, task: str) -> dict[int, dict]:
    """
    Load the aggregated fine-tuning results for one model, language and task.

    :param results_dir: Root results directory.
    :param model: Model name, e.g. "xlmr".
    :param language: Language code, e.g. "zul".
    :param task: Task name, e.g. "ner".
    :return: Mapping from checkpoint step to aggregated results, empty when missing.
    """
    path = (results_dir / f"{model}-large" / language / "aggregated"
            / f"{task_filename(task, language)}_aggregated.json")

    if not path.exists():
        logger.warning(f"No aggregated results at {path}, leaving that panel empty.")
        return {}

    with open(path) as f:
        raw = json.load(f)

    return {int(step): data for step, data in raw.items()}

def load_all(results_dir: Path, models: list[str], languages: list[str],
             tasks: list[str]) -> dict[tuple[str, str, str], dict[int, dict]]:
    """
    Load every aggregated result the requested grids need.

    :param results_dir: Root results directory.
    :param models: Models to load.
    :param languages: Languages to load.
    :param tasks: Tasks to load.
    :return: Mapping from (model, language, task) to aggregated results.
    """
    aggregated = {}

    for model in models:
        for language in languages:
            for task in tasks:
                results = load_aggregated(results_dir, model, language, task)
                if results:
                    aggregated[(model, language, task)] = results

    logger.info(f"Loaded aggregated results for {len(aggregated)} model/language/task cells.")
    return aggregated

def series(aggregated: dict[int, dict], steps: list[int], key: str,
           class_name: str | None = None) -> np.ndarray:
    """
    Read one metric series as a float array, with missing values left as NaN.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param steps: Checkpoint steps to read, in plotting order.
    :param key: Which aggregated field to read, e.g. "overall_mean".
    :param class_name: Class to read from a per-class field, if the field is per class.
    :return: Float array of the same length as steps.
    """
    values = []

    for step in steps:
        entry = aggregated[step].get(key)
        value = entry.get(class_name) if class_name is not None else entry
        values.append(np.nan if value is None else float(value))

    return np.array(values, dtype=float)

def plot_by_task(aggregated, models, languages, tasks, output_path: Path) -> None:
    """
    Plot overall F1 with rows for languages and columns for tasks, one line per
    model, so models can be read against each other within a task.

    :param aggregated: Mapping from (model, language, task) to aggregated results.
    :param models: Models to draw as lines.
    :param languages: Languages to draw as rows.
    :param tasks: Tasks to draw as columns.
    :param output_path: File path to save the figure to.
    """
    fig, axes = style.grid_figure(len(languages), len(tasks), row_labels=True)

    for row, language in enumerate(languages):
        for col, task in enumerate(tasks):
            ax = axes[row][col]

            for model in models:
                results = aggregated.get((model, language, task))
                if not results:
                    continue

                steps = sorted(results)
                ax.plot(
                    steps,
                    series(results, steps, "overall_mean"),
                    label=style.MODEL_DISPLAY_NAMES[model],
                    color=style.MODEL_COLOURS[model],
                )
                style.configure_step_axis(ax, steps)

    style.configure_value_axis(axes.flat)

    style.label_grid(
        axes,
        column_titles=[TASK_DISPLAY_NAMES[task] for task in tasks],
        xlabel=STEP_LABEL,
        ylabels=VALUE_LABEL,
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
    )
    logger.info(f"Saved downstream by-task grid to {output_path}")

def plot_overall(aggregated, models, languages, tasks, output_path: Path) -> None:
    """
    Plot overall F1 with rows for languages and columns for models, one line per
    task, so a single model's dynamics can be read across all of its tasks.

    :param aggregated: Mapping from (model, language, task) to aggregated results.
    :param models: Models to draw as columns.
    :param languages: Languages to draw as rows.
    :param tasks: Tasks to draw as lines.
    :param output_path: File path to save the figure to.
    """
    fig, axes = style.grid_figure(len(languages), len(models), row_labels=True)

    for row, language in enumerate(languages):
        for col, model in enumerate(models):
            ax = axes[row][col]

            for task in tasks:
                results = aggregated.get((model, language, task))
                if not results:
                    continue

                steps = sorted(results)
                ax.plot(
                    steps,
                    series(results, steps, "overall_mean"),
                    label=TASK_DISPLAY_NAMES[task],
                    color=style.TASK_COLOURS[task],
                )
                style.configure_step_axis(ax, steps)

    style.configure_value_axis(axes.flat)

    style.label_grid(
        axes,
        column_titles=[style.MODEL_DISPLAY_NAMES[model] for model in models],
        xlabel=STEP_LABEL,
        ylabels=VALUE_LABEL,
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
    )
    logger.info(f"Saved downstream overall grid to {output_path}")

def class_label(class_name: str) -> str:
    """
    Resolve a class to its display label, translating isiZulu NTC categories.

    :param class_name: Class as it appears in the aggregated results.
    :return: Label to show in the legend.
    """
    if class_name in CLASS_DISPLAY_NAMES:
        return CLASS_DISPLAY_NAMES[class_name]

    return class_name if class_name.isupper() else class_name.capitalize()

def group_of(class_name: str) -> str:
    """
    Resolve a part-of-speech tag to its coarse group.

    :param class_name: Tag as it appears in the aggregated results.
    :return: Group name, falling back to "Other" for unrecognised tags.
    """
    for group, tags in POS_GROUPS.items():
        if class_name.upper() in tags:
            return group

    return "Other"

def group_series(aggregated: dict[int, dict], steps: list[int], group: str) -> np.ndarray:
    """
    Average per-class F1 over the tags making up one coarse group.

    :param aggregated: Mapping from checkpoint step to aggregated results.
    :param steps: Checkpoint steps to read, in plotting order.
    :param group: Group to average over.
    :return: Float array of the same length as steps.
    """
    values = []

    for step in steps:
        scores = [
            score for class_name, score in aggregated[step]["per_class_mean"].items()
            if group_of(class_name) == group and score is not None
        ]
        values.append(float(np.mean(scores)) if scores else np.nan)

    return np.array(values, dtype=float)

def plot_grouped(aggregated, models, languages, task: str, output_path: Path) -> None:
    """
    Plot one task's per-class F1 averaged into coarse groups, with rows for
    languages and columns for models.

    :param aggregated: Mapping from (model, language, task) to aggregated results.
    :param models: Models to draw as columns.
    :param languages: Languages to draw as rows.
    :param task: Task whose classes are grouped.
    :param output_path: File path to save the figure to.
    """
    present = {
        group_of(class_name)
        for (_, language, cell_task), results in aggregated.items()
        if cell_task == task and language in languages
        for step_results in results.values()
        for class_name in step_results["per_class_mean"]
    }
    groups = [group for group in POS_GROUPS if group in present]

    if not groups:
        logger.warning(f"No per-class results for '{task}', skipping the grouped figure.")
        return

    colours = dict(zip(groups, style.categorical_colours(len(groups))))

    fig, axes = style.grid_figure(len(languages), len(models), row_labels=True)

    for row, language in enumerate(languages):
        for col, model in enumerate(models):
            ax = axes[row][col]

            results = aggregated.get((model, language, task))
            if not results:
                continue

            steps = sorted(results)

            for group in groups:
                values = group_series(results, steps, group)
                if np.isnan(values).all():
                    continue

                ax.plot(steps, values, label=group, color=colours[group])

            style.configure_step_axis(ax, steps)

    style.configure_value_axis(axes.flat)

    style.label_grid(
        axes,
        column_titles=[style.MODEL_DISPLAY_NAMES[model] for model in models],
        xlabel=STEP_LABEL,
        ylabels=VALUE_LABEL,
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
    )
    logger.info(f"Saved {TASK_DISPLAY_NAMES[task]} grouped grid to {output_path}")

def plot_per_class(aggregated, models, languages, task: str, output_path: Path) -> None:
    """
    Plot the per-class breakdown of one task, with rows for languages and columns
    for models, one line per class.

    :param aggregated: Mapping from (model, language, task) to aggregated results.
    :param models: Models to draw as columns.
    :param languages: Languages to draw as rows.
    :param task: Task whose classes are drawn as lines.
    :param output_path: File path to save the figure to.
    """
    class_names = sorted({
        class_name
        for (_, language, cell_task), results in aggregated.items()
        if cell_task == task and language in languages
        for step_results in results.values()
        for class_name in step_results["per_class_mean"]
    })

    if not class_names:
        logger.warning(f"No per-class results for '{task}', skipping that figure.")
        return

    # Colour by display label, so a category shared by both languages under
    # different names keeps one colour and one legend entry
    labels = sorted({class_label(class_name) for class_name in class_names})
    colours = dict(zip(labels, style.categorical_colours(len(labels))))

    fig, axes = style.grid_figure(len(languages), len(models), row_labels=True)

    for row, language in enumerate(languages):
        for col, model in enumerate(models):
            ax = axes[row][col]

            results = aggregated.get((model, language, task))
            if not results:
                continue

            steps = sorted(results)

            for class_name in class_names:
                values = series(results, steps, "per_class_mean", class_name)
                if np.isnan(values).all():
                    continue

                label = class_label(class_name)
                ax.plot(steps, values, label=label, color=colours[label])

            style.configure_step_axis(ax, steps)

    style.configure_value_axis(axes.flat)

    style.label_grid(
        axes,
        column_titles=[style.MODEL_DISPLAY_NAMES[model] for model in models],
        xlabel=STEP_LABEL,
        ylabels=VALUE_LABEL,
        fig=fig,
    )

    style.finalise_grid(
        fig, axes, output_path,
        row_labels=[style.LANGUAGE_DISPLAY_NAMES[language] for language in languages],
    )
    logger.info(f"Saved {TASK_DISPLAY_NAMES[task]} per-class grid to {output_path}")

def main() -> None:
    """Main entry point for plotting downstream results."""
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = load_all(results_dir, args.models, args.languages, args.tasks)

    if not aggregated:
        logger.error(f"No aggregated results found under {results_dir}")
        return

    plot_by_task(
        aggregated, args.models, args.languages, args.tasks,
        output_dir / "downstream_by_task.pdf"
    )

    plot_overall(
        aggregated, args.models, args.languages, args.tasks,
        output_dir / "downstream_overall.pdf"
    )

    for task in args.tasks:
        plot_per_class(
            aggregated, args.models, args.languages, task,
            output_dir / f"downstream_{task}_per_class.pdf"
        )

    if "pos" in args.tasks:
        plot_grouped(
            aggregated, args.models, args.languages, "pos",
            output_dir / "downstream_pos_grouped.pdf"
        )

if __name__ == "__main__":
    main()
