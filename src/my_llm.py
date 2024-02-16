from magentic import prompt
from magentic import chatprompt, SystemMessage, UserMessage
from magentic import OpenaiChatModel
from log10.load import log10
import openai

log10(openai)


# in order to make it work in 30 words, we need to prompt for less words
@prompt("Summarize the article in 25 words.\n Article: {article}")
def summarize_to_30_words(article: str) -> str:
    ...


sys_message_1 = SystemMessage("Summarize the article in 30 words.")
sys_message_2 = SystemMessage(
    "Summarize article into concise overview, focuing on the main points and conclusion."
)


@chatprompt(
    sys_message_1,
    UserMessage("Article: {article}"),
)
def summarize_with_sys_prompt_1(article: str) -> str:
    ...


@chatprompt(
    sys_message_2,
    UserMessage("Article: {article}"),
)
def summarize_with_sys_prompt_2(article: str) -> str:
    ...


def main():
    import jsonlines
    import os

    fill_summary = []
    with jsonlines.open("data.jsonl") as reader:
        for obj in reader:
            with OpenaiChatModel("gpt-4-0125-preview", temperature=0):
                summary = summarize_to_30_words(obj["article"])
                print(summary)
            if os.environ.get("LOG10_RECORD", None) == "1":
                obj["summary"] = summary
                fill_summary.append(obj)

    if os.environ.get("LOG10_RECORD", None) == "1":
        with jsonlines.open("data.jsonl", "w") as f:
            f.write_all(fill_summary)


if __name__ == "__main__":
    main()
