"""ADK SequentialAgent configuration for blog writing and publishing."""

import logging
import os
from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.apps import App
from google.adk.plugins.global_instruction_plugin import GlobalInstructionPlugin
from google.adk.plugins.logging_plugin import LoggingPlugin

from .callbacks import LoggingCallbacks, add_session_to_memory
from .prompt import (
    return_description_publisher,
    return_description_root,
    return_description_writer,
    return_global_instruction,
    return_instruction_publisher,
    return_instruction_writer,
)
from .tools import publish_blog_to_github, save_blog_content

logger = logging.getLogger(__name__)

logging_callbacks = LoggingCallbacks()

# Determine model configuration
model_name = os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash")
model: Any = model_name

# Explicitly use LiteLlm for OpenRouter or other provider-prefixed models
# that might not be auto-detected by ADK's registry.
if model_name.lower().startswith("openrouter/") or "/" in model_name:
    try:
        from google.adk.models import LiteLlm

        logger.info(f"Using LiteLlm for model: {model_name}")
        model = LiteLlm(model=model_name)
    except ImportError:
        logger.warning(
            "LiteLlm not available, falling back to string model name. "
            "OpenRouter models may not work."
        )

# Blog Writer Agent - responsible for writing and saving blog content
blog_writer_agent = LlmAgent(
    name="blog_writer",
    description=return_description_writer(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=logging_callbacks.after_agent,
    model=model,
    instruction=return_instruction_writer(),
    tools=[save_blog_content],
    before_model_callback=logging_callbacks.before_model,
    after_model_callback=logging_callbacks.after_model,
    before_tool_callback=logging_callbacks.before_tool,
    after_tool_callback=logging_callbacks.after_tool,
)

# Blog Publisher Agent - responsible for publishing the saved blog to GitHub
# Uses include_contents='none' to prevent receiving the blog content in history
blog_publisher_agent = LlmAgent(
    name="blog_publisher",
    description=return_description_publisher(),
    before_agent_callback=logging_callbacks.before_agent,
    after_agent_callback=[logging_callbacks.after_agent, add_session_to_memory],
    model=model,
    instruction=return_instruction_publisher(),
    tools=[publish_blog_to_github],
    include_contents="none",  # Don't include prior conversation history
    before_model_callback=logging_callbacks.before_model,
    after_model_callback=logging_callbacks.after_model,
    before_tool_callback=logging_callbacks.before_tool,
    after_tool_callback=logging_callbacks.after_tool,
)

# Sequential Agent - runs writer first, then publisher
# Both agents share the same session state and artifacts
root_agent = SequentialAgent(
    name="blog_agent",
    description=return_description_root(),
    sub_agents=[blog_writer_agent, blog_publisher_agent],
)

# Optional App configs explicitly set to None for template documentation
app = App(
    name="agent",
    root_agent=root_agent,
    plugins=[
        GlobalInstructionPlugin(return_global_instruction),
        LoggingPlugin(),
    ],
    events_compaction_config=None,
    context_cache_config=None,
    resumability_config=None,
)
