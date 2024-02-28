import pytest
import os
import sys

# Append the src directory to sys.path to make its modules available for import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from party_host import adjust_for_guest


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
