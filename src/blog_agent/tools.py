"""Custom tools for the LLM agent."""

import base64
import logging
import os
from io import BytesIO
from typing import Any

import requests
from google import genai
from google.adk.tools import ToolContext
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
BLOG_ARTIFACT_FILENAME = "blog_content.md"
BLOG_IMAGE_DIRECTORY_NAME = "images"
BLOG_IMAGE_MIME_TYPE = "image/png"
BLOG_IMAGE_PATH_TEMPLATE = "./images/{image_filename}"
DEFAULT_BLOG_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
BLOG_IMAGE_STYLE_GUIDANCE = (
    "Hand-drawn digital illustration, whiteboard doodle style, white background, "
    "black ink outlines. Mostly grayscale shading, but featuring one or two "
    "bright, vibrant accent colors to highlight the main action. Cute, "
    "whimsical comic style, tech humor."
)


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


def _get_blog_image_model() -> str:
    """Get the configured image generation model."""
    return os.getenv("BLOG_IMAGE_MODEL", DEFAULT_BLOG_IMAGE_MODEL)


def _build_blog_image_markdown_path(image_filename: str) -> str:
    """Build the markdown image path used inside the blog post."""
    return BLOG_IMAGE_PATH_TEMPLATE.format(image_filename=image_filename)


def _build_blog_image_repo_path(content_path: str, image_filename: str) -> str:
    """Build the repository path for the generated image asset."""
    return f"{content_path}/{BLOG_IMAGE_DIRECTORY_NAME}/{image_filename}"


def _build_image_generation_prompt(
    title: str, character_description: str, scene_description: str
) -> str:
    """Build a detailed prompt for the blog image generator."""
    prompt_sections = [
        f'Create one illustration for the blog post titled "{title}".',
        "Match this visual style exactly:",
        BLOG_IMAGE_STYLE_GUIDANCE,
        f"The image MUST feature this character: {character_description}",
        f"Scene: {character_description} is {scene_description}",
    ]
    return "\n\n".join(prompt_sections)


def _extract_response_text(response: Any) -> str:
    """Collect any text parts returned by the image model."""
    response_parts = getattr(response, "parts", None)
    if not response_parts:
        return ""

    text_segments: list[str] = []
    for part in response_parts:
        part_text = getattr(part, "text", None)
        if part_text:
            text_segments.append(part_text)

    return "\n".join(text_segments).strip()


def _extract_png_bytes(response: Any) -> bytes | None:
    """Extract the first generated image and normalize it as PNG bytes."""
    response_parts = getattr(response, "parts", None)
    if not response_parts:
        return None

    for part in response_parts:
        inline_data = getattr(part, "inline_data", None)
        if inline_data is None:
            continue

        raw_image_bytes = getattr(inline_data, "data", None)
        mime_type = getattr(inline_data, "mime_type", None)

        if isinstance(raw_image_bytes, bytes):
            if mime_type == BLOG_IMAGE_MIME_TYPE:
                return raw_image_bytes

            if isinstance(mime_type, str) and mime_type.startswith("image/"):
                source_image = Image.open(BytesIO(raw_image_bytes))
                image_buffer = BytesIO()
                source_image.save(image_buffer, format="PNG")
                return image_buffer.getvalue()

        generated_image = part.as_image()
        if generated_image is None:
            continue

        image_buffer = BytesIO()
        generated_image.save(image_buffer, "PNG")
        return image_buffer.getvalue()

    return None


def _get_existing_file_sha(
    *,
    base_url: str,
    headers: dict[str, str],
    branch_name: str,
    file_path: str,
) -> str | None:
    """Return the current SHA for a file on the target branch, if it exists."""
    response = requests.get(
        f"{base_url}/contents/{file_path}",
        headers=headers,
        params={"ref": branch_name},
        timeout=30,
    )

    if response.status_code == 404:
        return None

    if response.status_code != 200:
        raise GitHubError(
            message=f"Failed to inspect file: {response.status_code}",
            status_code=response.status_code,
            details=response.text,
        )

    file_details = response.json()
    file_sha = file_details.get("sha")
    if isinstance(file_sha, str):
        return file_sha

    return None


def _upload_file_to_github(
    *,
    base_url: str,
    headers: dict[str, str],
    branch_name: str,
    file_path: str,
    file_bytes: bytes,
    commit_message: str,
) -> None:
    """Create or update one file on the target branch."""
    existing_file_sha = _get_existing_file_sha(
        base_url=base_url,
        headers=headers,
        branch_name=branch_name,
        file_path=file_path,
    )

    file_request_body: dict[str, str] = {
        "message": commit_message,
        "content": base64.b64encode(file_bytes).decode("utf-8"),
        "branch": branch_name,
    }
    if existing_file_sha:
        file_request_body["sha"] = existing_file_sha

    response = requests.put(
        f"{base_url}/contents/{file_path}",
        headers=headers,
        json=file_request_body,
        timeout=30,
    )
    if response.status_code not in (200, 201):
        raise GitHubError(
            message=f"Failed to create file: {response.status_code}",
            status_code=response.status_code,
            details=response.text,
        )


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


async def generate_blog_image(
    tool_context: ToolContext,
    title: str,
    character_description: str,
    scene_description: str,
    alt_text: str,
    image_filename: str,
) -> dict[str, Any]:
    """Generate one blog image and save it as an artifact for publishing.
    Can be called multiple times to generate multiple images.
    """
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return {
            "status": "error",
            "message": "GEMINI_API_KEY not configured",
        }

    image_model = _get_blog_image_model()
    full_prompt = _build_image_generation_prompt(
        title=title,
        character_description=character_description,
        scene_description=scene_description,
    )

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model=image_model,
            contents=full_prompt,
        )

        image_bytes = _extract_png_bytes(response)
        if image_bytes is None:
            response_text = _extract_response_text(response)
            error_details = response_text or "The model did not return an image."
            return {
                "status": "error",
                "message": "Failed to generate blog image",
                "details": error_details,
            }

        artifact = types.Part.from_bytes(
            data=image_bytes,
            mime_type=BLOG_IMAGE_MIME_TYPE,
        )
        version = await tool_context.save_artifact(image_filename, artifact)

        image_markdown_path = _build_blog_image_markdown_path(image_filename)
        image_markdown = f"![{alt_text}]({image_markdown_path})"

        # Initialize the generated_images list in state if it doesn't exist
        if "generated_images" not in tool_context.state:
            tool_context.state["generated_images"] = []

        tool_context.state["generated_images"].append(
            {
                "image_filename": image_filename,
                "alt_text": alt_text,
                "image_markdown_path": image_markdown_path,
                "character_description": character_description,
                "scene_description": scene_description,
            }
        )

        logger.info(f"Saved blog image {image_filename} to artifact version {version}")

        return {
            "status": "success",
            "message": f"Blog image generated successfully (version {version})",
            "image_markdown": image_markdown,
            "image_path": image_markdown_path,
            "alt_text": alt_text,
            "image_filename": image_filename,
        }

    except Exception as e:
        logger.exception("Failed to generate blog image")
        return {
            "status": "error",
            "message": f"Failed to generate blog image: {e}",
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

        # 4. Upload the generated images if any exist
        generated_images = tool_context.state.get("generated_images", [])
        uploaded_image_paths = []

        for img_info in generated_images:
            image_filename = img_info["image_filename"]
            image_artifact = await tool_context.load_artifact(image_filename)
            image_blob = None if image_artifact is None else image_artifact.inline_data
            image_bytes = None if image_blob is None else image_blob.data

            if image_bytes:
                image_file_path = _build_blog_image_repo_path(
                    content_path, image_filename
                )
                image_commit_message = f"{commit_message} ({image_filename})"

                _upload_file_to_github(
                    base_url=base_url,
                    headers=headers,
                    branch_name=branch_name,
                    file_path=image_file_path,
                    file_bytes=image_bytes,
                    commit_message=image_commit_message,
                )
                uploaded_image_paths.append(image_file_path)

        # 5. Create or update the markdown file with exact content from artifact
        _upload_file_to_github(
            base_url=base_url,
            headers=headers,
            branch_name=branch_name,
            file_path=full_file_path,
            file_bytes=content.encode("utf-8"),
            commit_message=commit_message,
        )

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
            "image_file_paths": uploaded_image_paths,
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
