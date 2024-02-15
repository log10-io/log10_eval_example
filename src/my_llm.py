from magentic import prompt
from magentic import chatprompt, SystemMessage, UserMessage
from log10.load import log10
import openai

log10(openai)


@prompt("Summarize the article in 30 words.\n Article: {article}")
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
    article = """
Children of Time is a 2015 science fiction novel by author Adrian Tchaikovsky.

The work was praised by Financial Times for "tackling big themes—gods, messiahs, artificial intelligence, alienness—with brio."[1]

It was selected from a shortlist of six works[2] and a total pool of 113 books to be awarded the Arthur C. Clarke Award for best science fiction of the year in August 2016.[3][4] The director of the award program appraised the novel as having "universal scale and sense of wonder reminiscent of Clarke himself."[5]

In July 2017, the rights were optioned for a potential film adaptation.[6]

The next in the series, Children of Ruin, was published in 2019, followed by Children of Memory in 2022.[7]

In 2023 the series was awarded the Hugo Award for Best Series.
"""
    print(summarize_to_30_words(article))


if __name__ == "__main__":
    main()
