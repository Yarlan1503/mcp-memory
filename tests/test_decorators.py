import pytest
from mcp_memory._helpers import tool_error_handler


class TestToolErrorHandler:
    def test_success_passthrough(self):
        @tool_error_handler
        def good_tool(x):
            return {"result": x * 2}

        assert good_tool(5) == {"result": 10}

    def test_exception_returns_error(self):
        @tool_error_handler
        def bad_tool():
            raise ValueError("boom")

        result = bad_tool()
        assert "error" in result
        assert "boom" in result["error"]

    def test_preserves_function_name(self):
        @tool_error_handler
        def named_tool():
            pass

        assert named_tool.__name__ == "named_tool"
