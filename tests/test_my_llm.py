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
    sys_message_1,
    sys_message_2,
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

    df["article_long"] = df["article"]
    df["article_id"] = df["test_name"].str.split("_").str[-1]
    df["article"] = df.apply(lambda row: row["article_long"][:50] + f"... [more](#article-{row['article_id']})", axis=1)

    # get unique articles
    unique_articles = df["article_long"].unique()
    # import ipdb; ipdb.set_trace()
    long_articles_section = "## Long Articles\n\n"
    for i, text in enumerate(unique_articles):
        long_articles_section += f"### Article {i}\n"
        long_articles_section += f"{text}\n\n"

    selected_df = df[
        [
            "status",
            "article",
            "expected_summary",
            "output",
            "cos_sim",
            "num_words",
        ]
    ]
    table = tabulate(selected_df, headers="keys", tablefmt="pipe", showindex=True)

    output_dir = os.environ.get("OUTPUT_REPORT_DIR", "generated_reports")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/test_30_words_table.md", "w") as f:
        f.write("Generated from test_my_llm::test_pass_rate_of_30_words\n\n")
        f.write(f"## Test Pass Rate\nPass rate: {pass_rate * 100:.1f}%\n\n")
        f.write("## Detailed Results\n\n")
        f.write(table + "\n\n")
        f.write(long_articles_section)
    assert pass_rate > 0.66


@pytest.mark.repeat(3)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_with_sys_prompt_1(data, sample_idx, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_sys_prompt_1(article)
    metric = cosine_similarity(expected_summary, output)

    results_bag.test_name = f"test_summarize_sys_prompt_1_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.prompt = sys_message_1


@pytest.mark.repeat(3)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_with_sys_prompt_2(data, sample_idx, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_sys_prompt_2(article)
    metric = cosine_similarity(expected_summary, output)

    results_bag.test_name = f"test_summarize_sys_prompt_2_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = metric
    results_bag.prompt = sys_message_2


def test_compare_prompts_results(module_results_df):
    df = module_results_df[
        module_results_df["test_name"].str.contains("test_summarize_sys_prompt_")
    ]
    # save df to csv
    df.to_csv("module_results_df.csv", index=True)
    mean_1 = df[df["test_name"].str.contains("sys_prompt_1")]["cos_sim"].mean()
    std_1 = df[df["test_name"].str.contains("sys_prompt_1")]["cos_sim"].std()
    mean_2 = df[df["test_name"].str.contains("sys_prompt_2")]["cos_sim"].mean()
    std_2 = df[df["test_name"].str.contains("sys_prompt_2")]["cos_sim"].std()

    # remove new lines in output
    df["output"] = df["output"].str.replace("\n", " ")
    df["article_long"] = df["article"]
    df["article_id"] = df["test_name"].str.split("_").str[-1]
    df["article"] = df.apply(lambda row: row["article_long"][:50] + f"... [more](#article-{row['article_id']})", axis=1)

    unique_articles = df["article_long"].unique()
    long_articles_section = "## Long Articles\n\n"
    for i, text in enumerate(unique_articles):
        long_articles_section += f"### Article {i}\n"
        long_articles_section += f"{text}\n\n"

    # generate a markdown table for the results
    markdown_table = f"""
| Prompt | Mean Cosine Similarity | Std Dev |
|--------|------------------------| --------|
| sys_prompt_1 | {mean_1:.3f} | {std_1:.3f} |
| sys_prompt_2 | {mean_2:.3f} | {std_2:.3f} |
"""
    selected_df = df[
        [
            "prompt",
            "article",
            "expected_summary",
            "output",
            "cos_sim",
        ]
    ]
    table = tabulate(selected_df, headers="keys", tablefmt="pipe", floatfmt=".3f", showindex=False)
    output_dir = os.environ.get("OUTPUT_REPORT_DIR", "generated_reports")
    # check if the output directory exists and create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/prompt_comparison_report.md", "w") as f:
        f.write("generated from test_my_llm::test_compare_prompts_results\n\n")
        f.write("Compare the mean cosine similarity of the two system prompts\n\n")
        f.write(markdown_table + "\n\n")
        f.write("## Detailed Results\n\n")
        f.write(table + "\n\n")
        f.write(long_articles_section)
