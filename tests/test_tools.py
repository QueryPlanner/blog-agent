"""Unit tests for custom tools."""

import logging
from io import BytesIO
from unittest.mock import patch

import pytest

# Import mock classes from conftest
from conftest import MockState
from google.genai import types
from PIL import Image

from blog_agent.tools import (
    BLOG_ARTIFACT_FILENAME,
    generate_blog_image,
    save_blog_content,
)


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
        if filename == BLOG_ARTIFACT_FILENAME and self._artifact_content:
            return types.Part(text=self._artifact_content)
        return self._saved_artifacts.get(filename)


class FakeImagePart:
    """Fake image response part for image generation tests."""

    def __init__(self) -> None:
        self.text = None
        image = Image.new("RGB", (16, 16), color="white")
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        self.inline_data = types.Blob(
            data=image_buffer.getvalue(),
            mime_type="image/png",
        )

    def as_image(self) -> Image.Image:
        image = Image.new("RGB", (16, 16), color="white")
        return image


class FakeTextPart:
    """Fake text response part for image generation tests."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.inline_data = None


class FakeGenerateContentResponse:
    """Fake generate_content response with response parts."""

    def __init__(self, parts: list[FakeImagePart | FakeTextPart]) -> None:
        self.parts = parts


class FakeModels:
    """Fake models client for google-genai calls."""

    def __init__(self, response: FakeGenerateContentResponse) -> None:
        self._response = response

    def generate_content(
        self, model: str, contents: str
    ) -> FakeGenerateContentResponse:
        assert model
        assert contents
        return self._response


class FakeGenAiClient:
    """Fake google-genai client used in tests."""

    def __init__(self, response: FakeGenerateContentResponse) -> None:
        self.models = FakeModels(response)


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


class TestGenerateBlogImage:
    """Tests for the generate_blog_image function."""

    @pytest.mark.asyncio
    async def test_generate_blog_image_success(self) -> None:
        """Test that generate_blog_image saves a PNG artifact and state metadata."""
        state = MockState({})
        tool_context = MockArtifactToolContext(state=state)
        response = FakeGenerateContentResponse(parts=[FakeImagePart()])

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False),
            patch(
                "blog_agent.tools.genai.Client",
                return_value=FakeGenAiClient(response),
            ),
        ):
            result = await generate_blog_image(
                tool_context=tool_context,  # type: ignore[arg-type]
                title="A Test Blog",
                character_description="A young boy with messy red hair",
                scene_description="Fixing a server",
                alt_text="Hand-drawn illustration about reliable systems",
                image_filename="hero.png",
            )

        assert result["status"] == "success"
        assert result["image_markdown"] == (
            "![Hand-drawn illustration about reliable systems](./images/hero.png)"
        )
        assert (
            state["generated_images"][0]["image_markdown_path"] == "./images/hero.png"
        )
        assert (
            state["generated_images"][0]["alt_text"]
            == "Hand-drawn illustration about reliable systems"
        )

        image_artifact = tool_context._saved_artifacts["hero.png"]
        assert image_artifact.inline_data is not None
        assert image_artifact.inline_data.mime_type == "image/png"

        image_bytes = image_artifact.inline_data.data
        assert image_bytes is not None
        loaded_image = Image.open(BytesIO(image_bytes))
        assert loaded_image.format == "PNG"

    @pytest.mark.asyncio
    async def test_generate_blog_image_logs_success(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that generate_blog_image logs a success message."""
        caplog.set_level(logging.INFO)

        state = MockState({})
        tool_context = MockArtifactToolContext(state=state)
        response = FakeGenerateContentResponse(parts=[FakeImagePart()])

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False),
            patch(
                "blog_agent.tools.genai.Client",
                return_value=FakeGenAiClient(response),
            ),
        ):
            await generate_blog_image(
                tool_context=tool_context,  # type: ignore[arg-type]
                title="A Test Blog",
                character_description="A young boy with messy red hair",
                scene_description="Fixing a server",
                alt_text="Hand-drawn illustration about reliable systems",
                image_filename="hero.png",
            )

        assert "Saved blog image hero.png to artifact" in caplog.text
