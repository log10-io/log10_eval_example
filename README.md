# log10 llmeval example
An example on adding tests for your LLM application. 
* use pytest and its plugins
* compare different prompts
* use custom evaluation function
* summary and report
* gate deployment

...

## Installation and run
```
# Start your virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install project dependencies
pip install -r requirements.txt

# Run the tests
pytest tests -s
```

## Report
After running the tests, the test function `test_compare_prompts_results` will generate a report file `prompt_comparison_report.md`.

## Misc
The dataframe is saved to `module_results_df.csv`. 
