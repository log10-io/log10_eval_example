import pytest
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jsonlines
from tabulate import tabulate

from my_llm import (
    summarize_to_30_words,
    summarize_with_sys_prompt_1,
    summarize_with_sys_prompt_2,
    sys_message_1,
    sys_message_2,
)
from my_eval_metrics import cosine_distance, count_words
from report_utils import (
    filter_results_by_test_name,
    report_pass_rate,
    generate_results_table,
    generate_markdown_report,
)


@pytest.fixture
def article():
    return """
Children of Time is a 2015 science fiction novel by author Adrian Tchaikovsky.

It was selected from a shortlist of six works[2] and a total pool of 113 books to be awarded the Arthur C. Clarke Award for best science fiction of the year in August 2016.[3][4] The director of the award program appraised the novel as having "universal scale and sense of wonder reminiscent of Clarke himself."[5]

In July 2017, the rights were optioned for a potential film adaptation.[6]
"""


@pytest.fixture
def expected_summary():
    return """
\"Children of Time\" by Adrian Tchaikovsky, praised for addressing major themes, won the 2016 Arthur C. Clarke Award. It has sequels and might be adapted into a film. The series won the 2023 Hugo Award.
"""


@pytest.fixture
def data():
    filename = "data.jsonl"
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append((obj["article"], obj["summary"]))
    return data


@pytest.mark.repeat(3)
def test_simple(article, expected_summary, results_bag):
    output = summarize_to_30_words(article)
    metric = cosine_distance(expected_summary, output)

    results_bag.cos_sim = metric
    assert metric < 0.2


def test_mean_cosine_similarity(module_results_df: pd.DataFrame):
    print("Average cosine distance: ", module_results_df["cos_sim"].mean())
    print("Std dev of cosine distance: ", module_results_df["cos_sim"].std())

    assert module_results_df["cos_sim"].mean() < 0.2


@pytest.mark.repeat(2)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_to_30_words(data: list, sample_idx: int, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_to_30_words(article)
    metric = cosine_distance(expected_summary, output)
    num_words = count_words(output)

    results_bag.test_name = f"test_summarize_to_30_words_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.num_words = num_words

    assert num_words <= 30


def _prepare_long_string_section(dataframe: pd.DataFrame, column_name: str) -> str:
    dataframe[f"{column_name}_long"] = dataframe[column_name]
    dataframe[f"{column_name}_id"] = dataframe["test_name"].str.split("_").str[-1]
    dataframe[column_name] = dataframe.apply(
        lambda row: row[f"{column_name}_long"][:50] + f"... [more](#article-{row[f'{column_name}_id']})",
        axis=1,
    )

    # get unique items
    unique_items = dataframe[f"{column_name}_long"].unique()
    long_items_section = f"## {column_name.capitalize()} details\n\n"
    for i, text in enumerate(unique_items):
        long_items_section += f"### {column_name.capitalize()} {i}\n"
        long_items_section += f"{text}\n\n"

    return long_items_section


def test_pass_rate_of_30_words(module_results_df: pd.DataFrame):
    df = filter_results_by_test_name(module_results_df, "test_summarize_to_30_words")

    pass_rate, pass_rate_report_str = report_pass_rate(df)
    long_articles_section = _prepare_long_string_section(df, "article")
    selected_columns = [
        "status",
        "article",
        "expected_summary",
        "output",
        "cos_sim",
        "num_words",
    ]
    detailed_results = generate_results_table(df, selected_columns)

    generate_markdown_report(
        "test_summarize_to_30_words",
        [pass_rate_report_str, detailed_results, long_articles_section],
    )

    assert pass_rate > 0.66


@pytest.mark.repeat(3)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_with_sys_prompt_1(data: list, sample_idx: int, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_sys_prompt_1(article)
    metric = cosine_distance(expected_summary, output)

    results_bag.test_name = f"test_summarize_sys_prompt_1_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.prompt = sys_message_1


@pytest.mark.repeat(3)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_with_sys_prompt_2(data: list, sample_idx: int, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_sys_prompt_2(article)
    metric = cosine_distance(expected_summary, output)

    results_bag.test_name = f"test_summarize_sys_prompt_2_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.prompt = sys_message_2


def test_compare_prompts_results(module_results_df: pd.DataFrame):
    df = module_results_df[module_results_df["test_name"].str.contains("test_summarize_sys_prompt_")]
    # save df to csv
    df.to_csv("module_results_df.csv", index=True)

    # compare mean and std of cosine distance
    mean_1 = df[df["test_name"].str.contains("sys_prompt_1")]["cos_sim"].mean()
    std_1 = df[df["test_name"].str.contains("sys_prompt_1")]["cos_sim"].std()
    mean_2 = df[df["test_name"].str.contains("sys_prompt_2")]["cos_sim"].mean()
    std_2 = df[df["test_name"].str.contains("sys_prompt_2")]["cos_sim"].std()

    # Create a new dataframe
    summary_df = pd.DataFrame(
        {
            "Prompt": ["sys_prompt_1", "sys_prompt_2"],
            "Mean Cosine distance": [mean_1, mean_2],
            "Std Dev": [std_1, std_2],
        }
    )

    # plot summary_df to a bar chart and save to file
    plot_file = "generated_reports/test_compare_prompts_results.png"
    summary_df.plot(kind="bar", x="Prompt", y="Mean Cosine distance", yerr="Std Dev")
    plt.savefig(plot_file)

    markdown_table = tabulate(summary_df, headers="keys", tablefmt="pipe", showindex=False)
    prompt_comp_section = (
        "## Prompt Comparison\n\n" + markdown_table + "\n\n![Prompt Comparison](test_compare_prompts_results.png)"
    )

    # remove new lines in output
    df["output"] = df["output"].str.replace("\n", " ")

    long_articles_section = _prepare_long_string_section(df, "article")

    selected_columns = [
        "prompt",
        "article",
        "expected_summary",
        "output",
        "cos_sim",
    ]
    detailed_results = generate_results_table(df, selected_columns)
    generate_markdown_report(
        "test_compare_prompts_results",
        [prompt_comp_section, detailed_results, long_articles_section],
    )
