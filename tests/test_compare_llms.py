import pytest
import sys
import os
import time

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jsonlines
from compare_llms import summarize_with_anthropic, summarize_with_openai
from report_utils import generate_markdown_report, generate_results_table
from test_my_llm import _prepare_long_string_section


@pytest.fixture
def data():
    filename = "data.jsonl"
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append((obj["article"], obj["summary"]))
    return data


@pytest.mark.parametrize("model", ["gpt-3.5-turbo-0125", "gpt-4-0125-preview"])
@pytest.mark.parametrize("sample_idx", [0, 1, 2])
def test_openai_models(data: list, sample_idx: int, model: str, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_openai(model, article)

    results_bag.test_name = f"openai_{model}_{sample_idx}"
    results_bag.model = model
    results_bag.data_idx = sample_idx
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output


@pytest.mark.parametrize("model", ["claude-3-haiku-20240307", "claude-3-opus-20240229"])
@pytest.mark.parametrize("sample_idx", [0, 1, 2])
def test_anthropic_models(data: list, sample_idx: int, model: str, results_bag):
    article, expected_summary = data[sample_idx]
    output = summarize_with_anthropic(model, article)

    results_bag.test_name = f"anthropic_{model}_{sample_idx}"
    results_bag.model = model
    results_bag.data_idx = sample_idx
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output

def test_compare_llms(module_results_df):
    df = module_results_df
    df = df.sort_values("sample_idx")
    long_articles_section = _prepare_long_string_section(df, "article")
    selected_columns = [
        "article",
        "model",
        "duration_ms",
        "output",
    ]
    detailed_results = generate_results_table(df, selected_columns)
    generate_markdown_report("test_compare_llms", [detailed_results, long_articles_section])