"""Custom tools for the LLM agent."""

import base64
import logging
import os
from typing import Any

import requests
from google.adk.tools import ToolContext
from google.genai import types

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
BLOG_ARTIFACT_FILENAME = "blog_content.md"


class GitHubError(Exception):
    """Custom exception for GitHub API errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details


def _get_github_headers() -> dict[str, str]:
    """Get headers for GitHub API requests."""
    token = os.getenv("BLOG_GITHUB_TOKEN")
    if not token:
        raise GitHubError("BLOG_GITHUB_TOKEN not configured")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_repo_config() -> dict[str, str]:
    """Get repository configuration from environment variables."""
    return {
        "owner": os.getenv("BLOG_REPO_OWNER", "queryplanner"),
        "repo": os.getenv("BLOG_REPO_NAME", "blogs"),
        "content_path": os.getenv("BLOG_CONTENT_PATH", "src/data/blog"),
    }


async def save_blog_content(
    tool_context: ToolContext,
    content: str,
    title: str,
    slug: str,
) -> dict[str, Any]:
    """Save the blog content to session artifact for later publishing.

    This tool saves the complete blog post content to an artifact, which will be
    retrieved by the publisher agent. The content is stored exactly as provided
    and will not be modified during publishing.

    Args:
        tool_context: ADK ToolContext with access to artifact service
        content: The complete markdown content for the blog post including YAML
                 frontmatter
        title: The title of the blog post
        slug: The URL slug for the blog post (e.g., "my-awesome-post")

    Returns:
        A dictionary with status and confirmation message.
    """
    try:
        artifact = types.Part(text=content)
        version = await tool_context.save_artifact(BLOG_ARTIFACT_FILENAME, artifact)

        # Store metadata in state for the publisher agent to reference
        tool_context.state["title"] = title
        tool_context.state["slug"] = slug

        logger.info(f"Saved blog content to artifact version {version}")

        return {
            "status": "success",
            "message": f"Blog content saved successfully (version {version})",
            "title": title,
            "slug": slug,
        }

    except Exception as e:
        logger.exception("Failed to save blog content")
        return {
            "status": "error",
            "message": f"Failed to save blog content: {e}",
        }


async def publish_blog_to_github(
    tool_context: ToolContext,
    branch_name: str,
    file_name: str,
    commit_message: str,
    pr_title: str,
    pr_body: str,
) -> dict[str, Any]:
    """Publish the saved blog post by creating a branch, adding a file, and PR.

    This tool retrieves the blog content from the session artifact (saved by the
    writer agent) and publishes it to GitHub. The content is used exactly as saved,
    without any modifications.

    Args:
        tool_context: ADK ToolContext with access to artifact service
        branch_name: Name for the new branch (e.g., "blog/new-post-slug")
        file_name: Filename for the blog post (e.g., "my-post.md")
                   Will be placed under the configured content path
        commit_message: Commit message for the changes
        pr_title: Title for the pull request
        pr_body: Body/description for the pull request

    Returns:
        A dictionary with status, PR URL, and any error messages.
    """
    try:
        # Load the blog content from artifact
        artifact = await tool_context.load_artifact(BLOG_ARTIFACT_FILENAME)
        if artifact is None or artifact.text is None:
            return {
                "status": "error",
                "message": "No blog content found. Writer must save first.",
            }

        content = artifact.text
        logger.info(f"Loaded blog content from artifact ({len(content)} chars)")

        # Get repo config from environment
        repo_config = _get_repo_config()
        repo_owner = repo_config["owner"]
        repo_name = repo_config["repo"]
        content_path = repo_config["content_path"]
        full_file_path = f"{content_path}/{file_name}"

        headers = _get_github_headers()
        base_url = f"{GITHUB_API_URL}/repos/{repo_owner}/{repo_name}"

        # 1. Get default branch (usually main or master)
        resp = requests.get(f"{base_url}", headers=headers, timeout=30)
        if resp.status_code != 200:
            return {
                "status": "error",
                "message": f"Failed to fetch repo info: {resp.status_code}",
                "details": resp.text,
            }

        default_branch = resp.json().get("default_branch", "main")

        # 2. Get the SHA of the default branch
        resp = requests.get(
            f"{base_url}/git/ref/heads/{default_branch}",
            headers=headers,
            timeout=30,
        )
        if resp.status_code != 200:
            return {
                "status": "error",
                "message": f"Failed to get branch ref: {resp.status_code}",
                "details": resp.text,
            }
        base_sha = resp.json().get("object", {}).get("sha")
        if not base_sha:
            return {
                "status": "error",
                "message": "Could not get base SHA",
                "details": resp.text,
            }

        # 3. Create new branch
        resp = requests.post(
            f"{base_url}/git/refs",
            headers=headers,
            json={
                "ref": f"refs/heads/{branch_name}",
                "sha": base_sha,
            },
            timeout=30,
        )
        if resp.status_code == 422 and "already exists" in resp.text:
            logger.info(f"Branch {branch_name} already exists, proceeding.")
        elif resp.status_code != 201:
            return {
                "status": "error",
                "message": f"Failed to create branch: {resp.status_code}",
                "details": resp.text,
            }

        # 4. Check if file already exists to get its SHA (for updates)
        file_sha = None
        resp = requests.get(
            f"{base_url}/contents/{full_file_path}",
            headers=headers,
            params={"ref": branch_name},
            timeout=30,
        )
        if resp.status_code == 200:
            file_sha = resp.json().get("sha")

        # 5. Create or update the file with exact content from artifact
        file_data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch_name,
        }
        if file_sha:
            file_data["sha"] = file_sha

        resp = requests.put(
            f"{base_url}/contents/{full_file_path}",
            headers=headers,
            json=file_data,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            return {
                "status": "error",
                "message": f"Failed to create file: {resp.status_code}",
                "details": resp.text,
            }

        # 6. Create pull request
        resp = requests.post(
            f"{base_url}/pulls",
            headers=headers,
            json={
                "title": pr_title,
                "body": pr_body,
                "head": branch_name,
                "base": default_branch,
            },
            timeout=30,
        )

        pr_url = None
        if resp.status_code == 201:
            pr_url = resp.json().get("html_url")
        elif resp.status_code == 422 and "A pull request already exists" in resp.text:
            # Try to find the existing PR
            pr_list_resp = requests.get(
                f"{base_url}/pulls",
                headers=headers,
                params={"head": f"{repo_owner}:{branch_name}", "state": "open"},
                timeout=30,
            )
            if pr_list_resp.status_code == 200 and pr_list_resp.json():
                pr_url = pr_list_resp.json()[0].get("html_url")
                logger.info(f"Found existing PR: {pr_url}")
            else:
                return {
                    "status": "error",
                    "message": "PR already exists but could not find its URL",
                    "details": resp.text,
                }
        else:
            return {
                "status": "error",
                "message": f"Failed to create PR: {resp.status_code}",
                "details": resp.text,
            }

        logger.info(f"Successfully processed PR: {pr_url}")

        return {
            "status": "success",
            "message": "Blog post published successfully",
            "pr_url": pr_url,
            "branch": branch_name,
            "file_path": full_file_path,
        }

    except GitHubError as e:
        return {
            "status": "error",
            "message": str(e),
            "details": e.details,
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error during GitHub API call: {e}")
        return {
            "status": "error",
            "message": "Network error while communicating with GitHub",
            "details": str(e),
        }
    except Exception as e:
        logger.exception("Unexpected error in publish_blog_to_github")
        return {
            "status": "error",
            "message": "An unexpected error occurred",
            "details": str(e),
        }
