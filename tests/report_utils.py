import os
import pandas as pd
from tabulate import tabulate
from typing import List, Tuple


def filter_results_by_test_name(module_results_df: pd.DataFrame, test_name_prefix: str) -> pd.DataFrame:
    filtered_df = module_results_df[module_results_df["test_name"].str.contains(test_name_prefix)]
    return filtered_df


def report_pass_rate(dataframe: pd.DataFrame) -> Tuple[float, str]:
    pass_number = len(dataframe[dataframe["status"] == "passed"])
    total_runs = len(dataframe)
    pass_rate = pass_number / total_runs * 100
    report_str = f"## Test Pass Rate\n Pass rate {pass_rate:.1f}% ({pass_number}/{total_runs})"

    return pass_rate, report_str


def generate_results_table(dataframe: pd.DataFrame, column_list: list[str]) -> str:
    selected_df = dataframe[column_list]
    table = tabulate(selected_df, headers="keys", tablefmt="pipe", showindex=True)
    ret_str = f"## Test Results\n{table}"
    return ret_str


def generate_markdown_report(test_name: str, report_strings: List[str]):
    output_dir = os.environ.get("OUTPUT_REPORT_DIR", "generated_reports")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{test_name}.md", "w") as f:
        f.write(f"Generated from {test_name}.\n\n")
        for report_string in report_strings:
            f.write(report_string + "\n\n")
