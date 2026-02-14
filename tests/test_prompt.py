"""Unit tests for prompt definition functions."""

from datetime import date
from unittest.mock import patch

from conftest import MockReadonlyContext

from blog_agent.prompt import (
    return_description_publisher,
    return_description_root,
    return_description_writer,
    return_global_instruction,
    return_instruction_publisher,
    return_instruction_writer,
)


class TestReturnDescriptionRoot:
    """Tests for return_description_root function."""

    def test_returns_non_empty_string(self) -> None:
        """Test that function returns a non-empty description string."""
        description = return_description_root()

        assert isinstance(description, str)
        assert len(description) > 0

    def test_description_content(self) -> None:
        """Test that description is a non-empty string with meaningful content."""
        description = return_description_root()

        # Description should be a non-empty string (flexible for any blog_agent name)
        assert isinstance(description, str)
        assert len(description) > 0
        # Should contain at least some alphabetic characters
        assert any(c.isalpha() for c in description)

    def test_description_is_consistent(self) -> None:
        """Test that function returns the same description on multiple calls."""
        description1 = return_description_root()
        description2 = return_description_root()

        assert description1 == description2


class TestReturnDescriptionWriter:
    """Tests for return_description_writer function."""

    def test_returns_non_empty_string(self) -> None:
        """Test that function returns a non-empty description string."""
        description = return_description_writer()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "writer" in description.lower() or "blog" in description.lower()

    def test_description_is_consistent(self) -> None:
        """Test that function returns the same description on multiple calls."""
        description1 = return_description_writer()
        description2 = return_description_writer()

        assert description1 == description2


class TestReturnDescriptionPublisher:
    """Tests for return_description_publisher function."""

    def test_returns_non_empty_string(self) -> None:
        """Test that function returns a non-empty description string."""
        description = return_description_publisher()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "publish" in description.lower()

    def test_description_is_consistent(self) -> None:
        """Test that function returns the same description on multiple calls."""
        description1 = return_description_publisher()
        description2 = return_description_publisher()

        assert description1 == description2


class TestReturnInstructionWriter:
    """Tests for return_instruction_writer function."""

    def test_returns_non_empty_string(self) -> None:
        """Test that function returns a non-empty instruction string."""
        instruction = return_instruction_writer()

        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_instruction_contains_writer_guidance(self) -> None:
        """Test that instruction contains writer-specific guidance."""
        instruction = return_instruction_writer()

        # Should mention the save_blog_content tool
        assert "save_blog_content" in instruction
        assert "YAML" in instruction or "frontmatter" in instruction.lower()

    def test_instruction_is_consistent(self) -> None:
        """Test that function returns the same instruction on multiple calls."""
        instruction1 = return_instruction_writer()
        instruction2 = return_instruction_writer()

        assert instruction1 == instruction2


class TestReturnInstructionPublisher:
    """Tests for return_instruction_publisher function."""

    def test_returns_non_empty_string(self) -> None:
        """Test that function returns a non-empty instruction string."""
        instruction = return_instruction_publisher()

        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_instruction_contains_publisher_guidance(self) -> None:
        """Test that instruction contains publisher-specific guidance."""
        instruction = return_instruction_publisher()

        # Should mention the publish_blog_to_github tool
        assert "publish_blog_to_github" in instruction
        assert "GitHub" in instruction

    def test_instruction_is_consistent(self) -> None:
        """Test that function returns the same instruction on multiple calls."""
        instruction1 = return_instruction_publisher()
        instruction2 = return_instruction_publisher()

        assert instruction1 == instruction2


class TestReturnGlobalInstruction:
    """Tests for return_global_instruction InstructionProvider function."""

    def test_returns_string_with_context(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        """Test that InstructionProvider returns a string when given ReadonlyContext."""
        instruction = return_global_instruction(mock_readonly_context)  # type: ignore

        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_includes_current_date(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        """Test that instruction includes today's date dynamically."""
        instruction = return_global_instruction(mock_readonly_context)  # type: ignore
        today = str(date.today())

        assert today in instruction
        assert "date" in instruction.lower()

    def test_date_updates_dynamically(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        """Test that date updates when function is called on different days."""
        # Mock date.today() to return a specific date
        with patch("blog_agent.prompt.date") as mock_date:
            mock_date.today.return_value = date(2025, 1, 15)
            instruction1 = return_global_instruction(mock_readonly_context)  # type: ignore

            # Verify first date
            assert "2025-01-15" in instruction1

            # Change the mocked date
            mock_date.today.return_value = date(2025, 2, 20)
            instruction2 = return_global_instruction(mock_readonly_context)  # type: ignore

            # Verify second date
            assert "2025-02-20" in instruction2
            assert instruction1 != instruction2

    def test_accepts_readonly_context_parameter(self) -> None:
        """Test that function signature accepts ReadonlyContext as required by ADK."""
        # Create a context with state to ensure it's accessible if needed
        context = MockReadonlyContext(
            blog_agent_name="test_blog_agent",
            invocation_id="test-123",
            state={"user_id": "user_456", "preferences": {"theme": "dark"}},
        )

        # Function should execute without errors
        instruction = return_global_instruction(context)  # type: ignore

        # Verify it returns valid instruction
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_context_state_accessible_but_unused(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        """Test that context state is accessible but not currently used in instruction.

        This test documents that while the function receives ReadonlyContext with
        state access, the current implementation doesn't use state. This allows
        future enhancement to customize instructions based on session state.
        """
        # Create two contexts with different states
        context1 = MockReadonlyContext(state={"user_tier": "premium"})
        context2 = MockReadonlyContext(state={"user_tier": "free"})

        instruction1 = return_global_instruction(context1)  # type: ignore
        instruction2 = return_global_instruction(context2)  # type: ignore

        # Currently, instructions should be identical (state not used)
        # If future implementation uses state, this test will fail and should be updated
        assert instruction1 == instruction2

        # Verify state is accessible if needed in future
        assert context1.state["user_tier"] == "premium"
        assert context2.state["user_tier"] == "free"

    def test_instruction_format_consistency(
        self, mock_readonly_context: MockReadonlyContext
    ) -> None:
        """Test that instruction maintains consistent format across calls."""
        instruction1 = return_global_instruction(mock_readonly_context)  # type: ignore
        instruction2 = return_global_instruction(mock_readonly_context)  # type: ignore

        # Should be identical when called at same time (same date)
        assert instruction1 == instruction2

        # Should contain expected structure
        assert "\n" in instruction1  # Multi-line format
        assert "Today's date:" in instruction1
