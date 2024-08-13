import pytest
from pytest_check import check
import sys
import os
import pandas as pd

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import jsonlines
from log10.load import log10_session

from my_llm import (
    summarize_to_30_words,
)
from my_eval_metrics import cosine_distance, count_words
from report_utils import (
    filter_results_by_test_name,
    report_pass_rate,
)


@pytest.fixture
def session():
    with log10_session() as session:
        assert session.last_completion_id() is None, "No completion ID should be found."
        yield session


@pytest.fixture
def data():
    filename = "data.jsonl"
    data = []
    with jsonlines.open(filename) as reader:
        for obj in reader:
            data.append((obj["article"], obj["summary"]))
    return data


# @pytest.mark.repeat(2)
@pytest.mark.parametrize("sample_idx", range(3))
def test_summarize_to_30_words(data: list, sample_idx: int, results_bag, json_metadata, session):
    article, expected_summary = data[sample_idx]
    output = summarize_to_30_words(article)
    cos_distance = cosine_distance(expected_summary, output)
    num_words = count_words(output)

    results_bag.test_name = f"test_summarize_to_30_words_{sample_idx}"
    results_bag.article = article
    results_bag.expected_summary = expected_summary
    results_bag.output = output
    results_bag.cos_sim = cos_distance
    results_bag.num_words = num_words
    results_bag.log10_completion_url = session.last_completion_url()
    json_metadata["log10"] = {"last_completion_id": session.last_completion_url()}

    num_words_less_than_30 = num_words <= 30
    results_bag.num_words_less_than_30 = num_words_less_than_30

    cos_distance_less_than_02 = cos_distance < 0.2
    results_bag.cos_distance_less_than_02 = cos_distance_less_than_02

    assert num_words_less_than_30, f"Number of words is {num_words}, expected <= 30"
    assert cos_distance_less_than_02, f"Cosine distance is {cos_distance}, expected < 0.2"


def test_pass_rate_of_30_words(module_results_df: pd.DataFrame):
    #save module_results_df to csv
    module_results_df.to_csv("module_results_df_080724.csv", index=False)
    df = filter_results_by_test_name(module_results_df, "test_summarize_to_30_words")

    pass_rate, pass_rate_report_str = report_pass_rate(df)

    assert pass_rate > 0.66