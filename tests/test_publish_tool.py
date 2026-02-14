"""Unit tests for publish_blog_to_github tool."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests
from google.genai import types

from agent.tools import publish_blog_to_github


class MockState:
    def __init__(self, data=None):
        self.data = data or {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]


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


@pytest.fixture
def tool_context_with_artifact():
    """Create a tool context with a blog content artifact already saved."""
    return MockArtifactToolContext(
        state=MockState({"title": "Test Blog", "slug": "test-blog"}),
        artifact_content="# Test Blog\n\nThis is the blog content.",
    )


@pytest.fixture
def tool_context_no_artifact():
    """Create a tool context without any saved artifact."""
    return MockArtifactToolContext(state=MockState())


@pytest.fixture
def github_env(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "fake-token")
    monkeypatch.setenv("BLOG_REPO_OWNER", "test-owner")
    monkeypatch.setenv("BLOG_REPO_NAME", "test-repo")


@pytest.mark.asyncio
async def test_publish_blog_success(tool_context_with_artifact, github_env):
    """Test successful blog publication."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put:
        # Mock Step 1: Get repo info
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            # Mock Step 2: Get branch ref
            MagicMock(
                status_code=200, json=lambda: {"object": {"sha": "base-sha-123"}}
            ),
            # Mock Step 4: Check if file exists (returns 404)
            MagicMock(status_code=404),
        ]

        # Mock Step 3: Create branch
        mock_post.side_effect = [
            MagicMock(status_code=201),
            # Mock Step 6: Create PR
            MagicMock(
                status_code=201, json=lambda: {"html_url": "https://github.com/pr/1"}
            ),
        ]

        # Mock Step 5: Create file
        mock_put.return_value = MagicMock(status_code=201)

        result = await publish_blog_to_github(
            tool_context=tool_context_with_artifact,  # type: ignore
            branch_name="blog/test",
            file_name="test.md",
            commit_message="feat: add test post",
            pr_title="Add test post",
            pr_body="Body text",
        )

        assert result["status"] == "success"
        assert result["pr_url"] == "https://github.com/pr/1"
        assert mock_post.call_count == 2
        assert mock_put.call_count == 1


@pytest.mark.asyncio
async def test_publish_blog_with_custom_repo(tool_context_with_artifact, github_env):
    """Test blog publication with custom repo owner and name."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put:
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            MagicMock(
                status_code=200, json=lambda: {"object": {"sha": "base-sha-123"}}
            ),
            MagicMock(status_code=404),
        ]

        mock_post.side_effect = [
            MagicMock(status_code=201),
            MagicMock(
                status_code=201, json=lambda: {"html_url": "https://github.com/pr/2"}
            ),
        ]

        mock_put.return_value = MagicMock(status_code=201)

        result = await publish_blog_to_github(
            tool_context=tool_context_with_artifact,  # type: ignore
            branch_name="blog/test",
            file_name="test.md",
            commit_message="feat: add test post",
            pr_title="Add test post",
            pr_body="Body text",
            repo_owner="custom-owner",
            repo_name="custom-repo",
        )

        assert result["status"] == "success"
        # Verify custom repo was used
        call_args = mock_get.call_args_list[0]
        assert "custom-owner/custom-repo" in str(call_args)


@pytest.mark.asyncio
async def test_publish_blog_branch_exists_success(tool_context_with_artifact, github_env):
    """Test that it succeeds if the branch already exists."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put:
        # Mock Step 1 & 2
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            MagicMock(
                status_code=200, json=lambda: {"object": {"sha": "base-sha-123"}}
            ),
            # Step 4: Check if file exists
            MagicMock(status_code=404),
        ]

        # Mock Step 3: Create branch (fails with 422 - already exists)
        # Mock Step 6: Create PR (succeeds)
        mock_post.side_effect = [
            MagicMock(status_code=422, text='{"message": "Reference already exists"}'),
            MagicMock(
                status_code=201, json=lambda: {"html_url": "https://github.com/pr/1"}
            ),
        ]

        # Mock Step 5: Create file
        mock_put.return_value = MagicMock(status_code=201)

        result = await publish_blog_to_github(
            tool_context=tool_context_with_artifact,  # type: ignore
            branch_name="blog/test",
            file_name="test.md",
            commit_message="feat: add test post",
            pr_title="Add test post",
            pr_body="Body text",
        )

        assert result["status"] == "success"
        assert result["pr_url"] == "https://github.com/pr/1"


@pytest.mark.asyncio
async def test_publish_blog_pr_exists_success(tool_context_with_artifact, github_env):
    """Test that it succeeds if the PR already exists."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put:
        # Mock Step 1 & 2
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            MagicMock(
                status_code=200, json=lambda: {"object": {"sha": "base-sha-123"}}
            ),
            # Step 4: Check if file exists
            MagicMock(status_code=404),
            # Step 6 find PR: List PRs
            MagicMock(
                status_code=200,
                json=lambda: [{"html_url": "https://github.com/pr/existing"}],
            ),
        ]

        # Mock Step 3: Create branch
        # Mock Step 6: Create PR (fails with 422 - already exists)
        mock_post.side_effect = [
            MagicMock(status_code=201),
            MagicMock(
                status_code=422,
                text='{"message": "A pull request already exists for ..."}',
            ),
        ]

        # Mock Step 5: Create file
        mock_put.return_value = MagicMock(status_code=201)

        result = await publish_blog_to_github(
            tool_context=tool_context_with_artifact,  # type: ignore
            branch_name="blog/test",
            file_name="test.md",
            commit_message="feat: add test post",
            pr_title="Add test post",
            pr_body="Body text",
        )

        assert result["status"] == "success"
        assert result["pr_url"] == "https://github.com/pr/existing"


@pytest.mark.asyncio
async def test_publish_blog_no_token_error(tool_context_with_artifact, monkeypatch):
    """Test that it returns error if GITHUB_TOKEN is missing."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    result = await publish_blog_to_github(
        tool_context=tool_context_with_artifact,  # type: ignore
        branch_name="blog/test",
        file_name="test.md",
        commit_message="feat: add test post",
        pr_title="Add test post",
        pr_body="Body text",
    )

    assert result["status"] == "error"
    assert "GITHUB_TOKEN not configured" in result["message"]


@pytest.mark.asyncio
async def test_publish_blog_no_artifact_error(tool_context_no_artifact, github_env):
    """Test that it returns error if no blog content artifact exists."""
    result = await publish_blog_to_github(
        tool_context=tool_context_no_artifact,  # type: ignore
        branch_name="blog/test",
        file_name="test.md",
        commit_message="feat: add test post",
        pr_title="Add test post",
        pr_body="Body text",
    )

    assert result["status"] == "error"
    assert "No blog content found" in result["message"]


@pytest.mark.asyncio
async def test_publish_blog_uses_artifact_content(tool_context_with_artifact, github_env):
    """Test that the tool uses the exact content from the artifact."""
    with patch("requests.get") as mock_get, patch("requests.post") as mock_post, patch(
        "requests.put"
    ) as mock_put:
        # Setup mocks
        mock_get.side_effect = [
            MagicMock(status_code=200, json=lambda: {"default_branch": "main"}),
            MagicMock(
                status_code=200, json=lambda: {"object": {"sha": "base-sha-123"}}
            ),
            MagicMock(status_code=404),
        ]

        mock_post.side_effect = [
            MagicMock(status_code=201),
            MagicMock(
                status_code=201, json=lambda: {"html_url": "https://github.com/pr/1"}
            ),
        ]

        mock_put.return_value = MagicMock(status_code=201)

        await publish_blog_to_github(
            tool_context=tool_context_with_artifact,  # type: ignore
            branch_name="blog/test",
            file_name="test.md",
            commit_message="feat: add test post",
            pr_title="Add test post",
            pr_body="Body text",
        )

        # Verify the content from artifact was used
        call_args = mock_put.call_args
        content_arg = call_args.kwargs["json"]["content"]
        decoded_content = base64.b64decode(content_arg).decode("utf-8")

        # Should match the artifact content
        assert decoded_content == "# Test Blog\n\nThis is the blog content."
