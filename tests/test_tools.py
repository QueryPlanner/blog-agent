"""Unit tests for custom tools."""

import logging

import pytest

# Import mock classes from conftest
from conftest import MockState
from google.genai import types

from blog_agent.tools import save_blog_content


class MockArtifactToolContext:
    """Mock ToolContext with artifact support for testing."""

    def __init__(
        self,
        state: MockState | None = None,
        artifact_content: str | None = None,
    ) -> None:
        """Initialize mock context with state and artifact support."""
        self.state = state if state is not None else MockState()
        self._artifact_content = artifact_content
        self._saved_artifacts: dict[str, types.Part] = {}

    async def save_artifact(
        self,
        filename: str,
        artifact: types.Part,
        custom_metadata: dict | None = None,
    ) -> int:
        """Mock save_artifact that stores artifact."""
        self._saved_artifacts[filename] = artifact
        return len(self._saved_artifacts) - 1

    async def load_artifact(
        self,
        filename: str,
        version: int | None = None,
    ) -> types.Part | None:
        """Mock load_artifact that returns stored artifact."""
        if self._artifact_content:
            return types.Part(text=self._artifact_content)
        return self._saved_artifacts.get(filename)


class TestSaveBlogContent:
    """Tests for the save_blog_content function."""

    @pytest.mark.asyncio
    async def test_save_blog_content_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that save_blog_content saves content to artifact."""
        caplog.set_level(logging.INFO)

        # Create mock context
        state = MockState({})
        tool_context = MockArtifactToolContext(state=state)

        # Execute tool
        result = await save_blog_content(
            tool_context=tool_context,  # type: ignore
            content="# My Blog\n\nThis is my blog content.",
            title="My First Blog",
            slug="my-first-blog",
        )

        # Verify return value
        assert result["status"] == "success"
        assert "saved successfully" in result["message"]
        assert result["title"] == "My First Blog"
        assert result["slug"] == "my-first-blog"

        # Verify state was updated
        assert state["title"] == "My First Blog"
        assert state["slug"] == "my-first-blog"

    @pytest.mark.asyncio
    async def test_save_blog_content_with_yaml_frontmatter(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test saving blog content with YAML frontmatter."""
        caplog.set_level(logging.INFO)

        state = MockState({})
        tool_context = MockArtifactToolContext(state=state)

        content = """---
title: Test Blog
author: Test Author
pubDatetime: 2025-01-15T10:00:00Z
slug: test-blog
tags:
  - test
---

# Test Blog

This is test content.
"""

        result = await save_blog_content(
            tool_context=tool_context,  # type: ignore
            content=content,
            title="Test Blog",
            slug="test-blog",
        )

        assert result["status"] == "success"
        assert state["title"] == "Test Blog"

    @pytest.mark.asyncio
    async def test_save_blog_content_logs_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that save_blog_content logs success message."""
        caplog.set_level(logging.INFO)

        state = MockState({})
        tool_context = MockArtifactToolContext(state=state)

        await save_blog_content(
            tool_context=tool_context,  # type: ignore
            content="Content",
            title="Title",
            slug="slug",
        )

        assert "Saved blog content to artifact" in caplog.text
