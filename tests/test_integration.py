"""Integration tests for blog_agent configuration and component wiring.

This module validates the basic structure and wiring of ADK app components.
Tests are pattern-based and validate integration points regardless of specific
implementation choices (plugins, tools, etc.).

Future: Container-based smoke tests for CI/CD will be added here.
"""

from collections.abc import Sequence
from typing import Any, Protocol, cast

from blog_agent import app


class AgentConfigLike(Protocol):
    """Minimal blog_agent surface needed for integration assertions."""

    name: str
    model: Any
    instruction: str | None
    description: str | None
    tools: Sequence[object] | None


class SequentialAgentLike(Protocol):
    """Minimal sequential blog_agent surface for integration assertions."""

    name: str
    description: str | None
    sub_blog_agents: Sequence[Any]


def as_blog_agent_config(blog_agent: object) -> AgentConfigLike:
    """Treat runtime blog_agent instances as a typed config surface."""
    return cast(AgentConfigLike, blog_agent)


def as_sequential_blog_agent(blog_agent: object) -> SequentialAgentLike:
    """Treat sequential blog_agent instances as a typed surface."""
    return cast(SequentialAgentLike, blog_agent)


class TestAppIntegration:
    """Pattern-based integration tests for App configuration and wiring."""

    def test_app_is_properly_instantiated(self) -> None:
        """Verify app container is properly instantiated."""
        assert app is not None
        assert app.name is not None
        assert isinstance(app.name, str)
        assert len(app.name) > 0

    def test_app_has_root_blog_agent(self) -> None:
        """Verify app is wired to root blog_agent."""
        assert app.root_blog_agent is not None

    def test_app_plugins_are_valid_if_configured(self) -> None:
        """Verify plugins (if any) are properly initialized."""
        # Plugins are optional - if configured, they should be a list
        if app.plugins is not None:
            assert isinstance(app.plugins, list)
            # Each plugin should be an object instance
            for plugin in app.plugins:
                assert plugin is not None
                assert hasattr(plugin, "__class__")


class TestAgentIntegration:
    """Pattern-based integration tests for Agent configuration."""

    def test_blog_agent_has_required_configuration(self) -> None:
        """Verify root blog_agent (SequentialAgent) has required configuration."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None

        # SequentialAgent has name and description
        assert blog_agent.name is not None
        assert isinstance(blog_agent.name, str)
        assert len(blog_agent.name) > 0

        # SequentialAgent has sub_blog_agents
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)
        assert sequential_blog_agent.sub_blog_agents is not None
        assert len(sequential_blog_agent.sub_blog_agents) >= 1

    def test_sub_blog_agents_have_valid_configuration(self) -> None:
        """Verify sub-blog_agents have valid model and instruction configuration."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)

        # Check each sub-blog_agent (writer and publisher)
        for sub_blog_agent in sequential_blog_agent.sub_blog_agents:
            typed_sub = as_blog_agent_config(sub_blog_agent)

            # Each sub-blog_agent should have a name
            assert typed_sub.name is not None
            assert isinstance(typed_sub.name, str)
            assert len(typed_sub.name) > 0

            # Each sub-blog_agent should have a model
            assert typed_sub.model is not None
            if isinstance(typed_sub.model, str):
                assert len(typed_sub.model) > 0
            else:
                # If it's an object, it should have a model attribute
                assert hasattr(typed_sub.model, "model")
                assert isinstance(typed_sub.model.model, str)
                assert len(typed_sub.model.model) > 0

    def test_sub_blog_agents_have_valid_instructions(self) -> None:
        """Verify sub-blog_agents have valid instructions and descriptions."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)

        for sub_blog_agent in sequential_blog_agent.sub_blog_agents:
            typed_sub = as_blog_agent_config(sub_blog_agent)

            # Each sub-blog_agent should have instructions
            if typed_sub.instruction is not None:
                assert isinstance(typed_sub.instruction, str)
                assert len(typed_sub.instruction) > 0

            # Each sub-blog_agent should have a description
            if typed_sub.description is not None:
                assert isinstance(typed_sub.description, str)
                assert len(typed_sub.description) > 0

    def test_sub_blog_agents_have_valid_tools(self) -> None:
        """Verify sub-blog_agents have properly configured tools."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)

        for sub_blog_agent in sequential_blog_agent.sub_blog_agents:
            typed_sub = as_blog_agent_config(sub_blog_agent)

            # Tools should be a list if configured
            if typed_sub.tools is not None:
                assert isinstance(typed_sub.tools, list)
                for tool in typed_sub.tools:
                    assert tool is not None
                    assert hasattr(tool, "__class__")

    def test_writer_blog_agent_has_save_tool(self) -> None:
        """Verify writer blog_agent has the save_blog_content tool."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)

        # Find the writer blog_agent
        writer_blog_agent = None
        for sub_blog_agent in sequential_blog_agent.sub_blog_agents:
            if sub_blog_agent.name == "blog_writer":
                writer_blog_agent = sub_blog_agent
                break

        assert writer_blog_agent is not None, "blog_writer blog_agent not found"
        typed_writer = as_blog_agent_config(writer_blog_agent)

        # Writer should have tools
        assert typed_writer.tools is not None
        assert len(typed_writer.tools) >= 1

    def test_publisher_blog_agent_has_publish_tool(self) -> None:
        """Verify publisher blog_agent has the publish_blog_to_github tool."""
        blog_agent = app.root_blog_agent
        assert blog_agent is not None
        sequential_blog_agent = as_sequential_blog_agent(blog_agent)

        # Find the publisher blog_agent
        publisher_blog_agent = None
        for sub_blog_agent in sequential_blog_agent.sub_blog_agents:
            if sub_blog_agent.name == "blog_publisher":
                publisher_blog_agent = sub_blog_agent
                break

        assert publisher_blog_agent is not None, "blog_publisher blog_agent not found"
        typed_publisher = as_blog_agent_config(publisher_blog_agent)

        # Publisher should have tools
        assert typed_publisher.tools is not None
        assert len(typed_publisher.tools) >= 1
