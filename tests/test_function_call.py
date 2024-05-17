import pytest
import os
import sys

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from party_host import adjust_for_guest
from get_weather import weather_of


@pytest.mark.parametrize(
    "statement, expected",
    [
        (
            "I can't hear the music",
            {"function": "change_music_volume", "arguments": {"increment": 2}, "final": "Music volume change: 2"},
        ),
        (
            "Do you have any more food?",
            {"function": "order_food", "arguments": {"food": "pizza", "amount": 3}, "final": "Ordered 3 pizza"},
        ),
    ],
)
def test_adjust_for_guest(statement, expected):
    func = adjust_for_guest(statement)
    output = func()
    function_name_list = [func.__name__ for func in adjust_for_guest.functions]
    assert func.function.__name__ in function_name_list
    assert func.function.__name__ == expected["function"]
    assert func.arguments == expected["arguments"]
    assert output == expected["final"]


def test_weather_of():
    final_response, intermediate_tool_call = weather_of("HQ of OpenAI")

    # check intermediate tool call is correct
    assert len(intermediate_tool_call) == 1
    assert intermediate_tool_call[0].function.name == "get_current_weather"
    assert intermediate_tool_call[0].function.arguments == '{"location":"San Francisco, CA"}'

    # check final response is correct
    assert "San Francisco" in final_response and "72" in final_response
