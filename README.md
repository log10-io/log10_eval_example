# log10 llmeval example
An example on adding tests for your LLM application. 
* use pytest and its plugins
* compare different prompts
* use custom evaluation function
* summary and report
* gate deployment

...

## Installation
```
# Start your virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

## Run tests and generate reports
Run all tests `pytest tests -s -v`

After running the tests, some tests generate reports in markdown format, saved in [`./generated_reports`](./generated_reports/).
* Test `test_pass_rate_of_30_words` generates the report file [`test_summarize_to_30_words.md`](./generated_reports/test_summarize_to_30_words.md).
* Test `test_compare_prompts_results` generates a report file [`test_compare_prompts_results.md`](./generated_reports/test_compare_prompts_results.md).

You can run selected test to generate each reports: 
```
# test_summarize_to_30_words.md
pytest tests/test_my_llm.py -k "test_summarize_to_30_words or test_pass_rate_of_30_words" -s

# test_compare_prompts_results.md
pytest tests/test_my_llm.py -k "test_summarize_with_sys_prompt_1 or test_summarize_with_sys_prompt_2 or test_compare_prompts_results" -s
```

## Misc
The dataframe is saved to `module_results_df.csv`. 
