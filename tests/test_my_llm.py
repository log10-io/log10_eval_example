import pytest
import sys
import os

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from my_llm import summerize_to_30_words
from my_eval_metrics import cosine_similarity


@pytest.mark.repeat(3)
def test_summerize(results_bag):
    article = """
Children of Time is a 2015 science fiction novel by author Adrian Tchaikovsky.

The work was praised by Financial Times for "tackling big themes—gods, messiahs, artificial intelligence, alienness—with brio."[1]

It was selected from a shortlist of six works[2] and a total pool of 113 books to be awarded the Arthur C. Clarke Award for best science fiction of the year in August 2016.[3][4] The director of the award program appraised the novel as having "universal scale and sense of wonder reminiscent of Clarke himself."[5]

In July 2017, the rights were optioned for a potential film adaptation.[6]

The next in the series, Children of Ruin, was published in 2019, followed by Children of Memory in 2022.[7]

In 2023 the series was awarded the Hugo Award for Best Series.
"""
    expected_summary = """
"Children of Time" is a science fiction novel praised for tackling big themes and was awarded the Arthur C. Clarke Award. It has been optioned for a potential film adaptation. The series also received the Hugo Award.
"""
    output = summerize_to_30_words(article)
    metric = cosine_similarity(expected_summary, output)
    results_bag.cos_sim = metric

def test_mean_cosine_similarity(module_results_df):
    print(module_results_df)
    print("Average cosine similarity: " module_results_df["cos_sim"].mean())