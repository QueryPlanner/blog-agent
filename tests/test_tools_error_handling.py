"""Unit tests for error handling in custom tools."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.genai import types

from agent.tools import (
    GitHubError,
    _get_github_headers,
    publish_blog_to_github,
    save_blog_content,
)


class MockState:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


class MockArtifactToolContext:
    """Mock ToolContext with artifact support for testing."""

    def __init__(
        self,
        state: MockState | None = None,
        artifact_content: str | None = None,
    ) -> None:
        self.state = state if state is not None else MockState()
        self._artifact_content = artifact_content
        self._saved_artifacts: dict[str, types.Part] = {}
        self.should_fail_save = False

    async def save_artifact(
        self,
        filename: str,
        artifact: types.Part,
        custom_metadata: dict[str, Any] | None = None,
    ) -> int:
        if self.should_fail_save:
            raise RuntimeError("Artifact service down")
        self._saved_artifacts[filename] = artifact
        return len(self._saved_artifacts) - 1

    async def load_artifact(
        self,
        filename: str,
        version: int | None = None,
    ) -> types.Part | None:
        if self._artifact_content:
            return types.Part(text=self._artifact_content)
        return self._saved_artifacts.get(filename)


@pytest.fixture
def tool_context() -> MockArtifactToolContext:
    return MockArtifactToolContext(state=MockState())


class TestSaveBlogContentErrors:
    @pytest.mark.asyncio
    async def test_save_blog_content_exception(
        self, tool_context: MockArtifactToolContext
    ) -> None:
        """Test handling of exceptions during save."""
        tool_context.should_fail_save = True

        result = await save_blog_content(
            tool_context=tool_context,  # type: ignore[arg-type]
            content="test",
            title="test",
            slug="test",
        )

        assert result["status"] == "error"
        assert "Failed to save blog content" in result["message"]
        assert "Artifact service down" in result["message"]


class TestPublishBlogErrors:
    def test_get_github_headers_missing_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test _get_github_headers raises error when token missing."""
        monkeypatch.delenv("BLOG_GITHUB_TOKEN", raising=False)
        with pytest.raises(GitHubError, match="BLOG_GITHUB_TOKEN not configured"):
            _get_github_headers()

    @pytest.mark.asyncio
    async def test_publish_blog_repo_info_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when fetching repo info fails."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with patch("requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404, text="Not Found")

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Failed to fetch repo info" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_branch_ref_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when fetching branch ref fails."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=404, text="Ref not found"),
            ]

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Failed to get branch ref" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_no_sha_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when SHA is missing in response."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with patch("requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {}}),  # No SHA
            ]

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Could not get base SHA" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_create_branch_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when creating branch fails."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
        ):
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {"sha": "sha123"}}),
            ]
            mock_post.return_value = MagicMock(status_code=403, text="Forbidden")

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Failed to create branch" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_create_file_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when creating file fails."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.put") as mock_put,
        ):
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {"sha": "sha123"}}),
                MagicMock(status_code=404),  # File doesn't exist
            ]
            mock_post.return_value = MagicMock(status_code=201)
            mock_put.return_value = MagicMock(status_code=500, text="Server Error")

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Failed to create file" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_create_pr_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when creating PR fails."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.put") as mock_put,
        ):
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {"sha": "sha123"}}),
                MagicMock(status_code=404),  # File doesn't exist
            ]
            mock_post.side_effect = [
                MagicMock(status_code=201),  # Create branch
                MagicMock(status_code=500, text="PR Error"),  # Create PR
            ]
            mock_put.return_value = MagicMock(status_code=201)

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Failed to create PR" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_pr_exists_but_lookup_fails(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test error when PR exists but cannot be found."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.put") as mock_put,
        ):
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {"sha": "sha123"}}),
                MagicMock(status_code=404),  # File doesn't exist
                MagicMock(status_code=200, json=lambda: []),  # Empty PR list
            ]
            mock_post.side_effect = [
                MagicMock(status_code=201),  # Create branch
                MagicMock(
                    status_code=422,
                    text="A pull request already exists",
                ),  # Create PR fails
            ]
            mock_put.return_value = MagicMock(status_code=201)

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "PR already exists but could not find its URL" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_update_existing_file(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test updating an existing file."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch("requests.put") as mock_put,
        ):
            mock_get.side_effect = [
                MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
                MagicMock(status_code=200, json=lambda: {"object": {"sha": "sha123"}}),
                # Check if file exists - returns 200 and SHA
                MagicMock(status_code=200, json=lambda: {"sha": "existing_file_sha"}),
            ]
            mock_post.side_effect = [
                MagicMock(status_code=201),  # Create branch
                MagicMock(
                    status_code=201, json=lambda: {"html_url": "http://pr"}
                ),  # Create PR
            ]
            mock_put.return_value = MagicMock(status_code=200)

            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "success"
            # Verify PUT was called with SHA
            call_args = mock_put.call_args
            assert call_args.kwargs["json"]["sha"] == "existing_file_sha"

    @pytest.mark.asyncio
    async def test_publish_blog_network_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of network errors."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with patch(
            "requests.get",
            side_effect=import_requests().exceptions.RequestException(
                "Connection error"
            ),
        ):
            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "Network error" in result["message"]

    @pytest.mark.asyncio
    async def test_publish_blog_unexpected_error(
        self, tool_context: MockArtifactToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test handling of unexpected errors."""
        tool_context._artifact_content = "content"
        monkeypatch.setenv("BLOG_GITHUB_TOKEN", "token")

        with patch("requests.get", side_effect=Exception("Boom")):
            result = await publish_blog_to_github(
                tool_context=tool_context,  # type: ignore[arg-type]
                branch_name="branch",
                file_name="file.md",
                commit_message="msg",
                pr_title="title",
                pr_body="body",
            )

            assert result["status"] == "error"
            assert "An unexpected error occurred" in result["message"]


def import_requests() -> Any:
    """Helper to import requests safely."""
    import requests

    return requests
