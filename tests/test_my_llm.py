import pytest
import sys
import os

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jsonlines
from tabulate import tabulate

from my_llm import (
    summarize_to_30_words,
    summarize_with_sys_prompt_1,
    summarize_with_sys_prompt_2,
)
from my_eval_metrics import cosine_similarity, count_words


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
    metric = cosine_similarity(expected_summary, output)

    results_bag.cos_sim = metric
    assert metric < 0.2


def test_mean_cosine_similarity(module_results_df):
    print("Average cosine similarity: ", module_results_df["cos_sim"].mean())
    print("Std dev of cosine similarity: ", module_results_df["cos_sim"].std())

    assert module_results_df["cos_sim"].mean() < 0.2


@pytest.mark.repeat(2)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_to_30_words(data, sample_idx, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_to_30_words(article)
    metric = cosine_similarity(expected_summary, output)
    num_words = count_words(output)

    results_bag.test_name = f"test_summarize_to_30_words_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.num_words = num_words

    assert num_words <= 30


def test_pass_rate_of_30_words(module_results_df):
    df = module_results_df[
        module_results_df["test_name"].str.contains("test_summarize_to_30_words")
    ]
    pass_rate = len(df[df["status"] == "passed"]) / len(df)
    selected_df = df[
        [
            "test_name",
            "status",
            "article",
            "expected_summary",
            "output",
            "cos_sim",
            "num_words",
        ]
    ]
    table = tabulate(selected_df, headers="keys", tablefmt="pipe", showindex=True)
    with open("test_30_words_table.md", "w") as f:
        f.write("Generated from test_my_llm::test_pass_rate_of_30_words\n\n")
        f.write(table + "\n\n")
        f.write(f"## Test Pass Rate\nPass rate: {pass_rate * 100:.1f}%")
    assert pass_rate > 0.66


@pytest.mark.repeat(3)
def test_summarize_with_sys_prompt_1(article, expected_summary, results_bag):
    output = summarize_with_sys_prompt_1(article)
    metric = cosine_similarity(expected_summary, output)
    results_bag.prompt = "sys_prompt_1"
    results_bag.cos_sim = metric


@pytest.mark.repeat(3)
def test_summarize_with_sys_prompt_2(article, expected_summary, results_bag):
    output = summarize_with_sys_prompt_2(article)
    metric = cosine_similarity(expected_summary, output)
    results_bag.prompt = "sys_prompt_2"
    results_bag.cos_sim = metric


def test_compare_prompts_results(module_results_df):
    df = module_results_df
    # save df to csv
    df.to_csv("module_results_df.csv", index=True)
    mean_1 = df[df["prompt"] == "sys_prompt_1"]["cos_sim"].mean()
    mean_2 = df[df["prompt"] == "sys_prompt_2"]["cos_sim"].mean()
    # generate a markdown table for the results
    markdown_table = f""" (This is generated from test test_my_llm::test_compare_prompts_results)

Compare the mean cosine similarity of the two system prompts:

| Prompt | Mean Cosine Similarity |
|--------|------------------------|
| sys_prompt_1 | {mean_1:.3f} |
| sys_prompt_2 | {mean_2:.3f} |
"""
    with open("prompt_comparison_report.md", "w") as f:
        f.write(markdown_table)
