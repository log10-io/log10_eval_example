# this will also log to log10.
# if you don't want to log, use 
from openai import OpenAI
from anthropic import Anthropic
# from log10.load import Anthropic, OpenAI

SYSTEM_PROMPT = "Summarize the article in 30 words."

def summarize_with_openai(gpt_model: str, article: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def summarize_with_anthropic(claude_model: str, article: str) -> str:
    client = Anthropic()
    response = client.messages.create(
        model=claude_model,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": article},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return response.content[0].text


