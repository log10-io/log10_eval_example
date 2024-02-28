# example from Magentic https://magentic.dev/chat-prompting/#functioncall
from magentic import (
    chatprompt,
    AssistantMessage,
    FunctionCall,
    UserMessage,
    SystemMessage,
)
from log10.load import log10
import openai

log10(openai)


def change_music_volume(increment: int):
    """Change music volume level. Min 1, max 10."""
    return f"Music volume change: {increment}"


def order_food(food: str, amount: int):
    """Order food."""
    return f"Ordered {amount} {food}"


@chatprompt(
    SystemMessage(
        "You are hosting a party and must keep the guests happy." "Call functions as needed. Do not respond directly."
    ),
    UserMessage("It's pretty loud in here!"),
    AssistantMessage(FunctionCall(change_music_volume, -2)),
    UserMessage("{request}"),
    functions=[change_music_volume, order_food],
)
def adjust_for_guest(request: str) -> FunctionCall[None]:
    ...
