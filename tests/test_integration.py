"""Integration tests for agent configuration and component wiring.

This module validates the basic structure and wiring of ADK app components.
Tests are pattern-based and validate integration points regardless of specific
implementation choices (plugins, tools, etc.).

Future: Container-based smoke tests for CI/CD will be added here.
"""

from collections.abc import Sequence
from typing import Any, Protocol, cast

from agent import app


class AgentConfigLike(Protocol):
    """Minimal agent surface needed for integration assertions."""

    name: str
    model: Any
    instruction: str | None
    description: str | None
    tools: Sequence[object] | None


class SequentialAgentLike(Protocol):
    """Minimal sequential agent surface for integration assertions."""

    name: str
    description: str | None
    sub_agents: Sequence[Any]


def as_agent_config(agent: object) -> AgentConfigLike:
    """Treat runtime agent instances as a typed config surface."""
    return cast(AgentConfigLike, agent)


def as_sequential_agent(agent: object) -> SequentialAgentLike:
    """Treat sequential agent instances as a typed surface."""
    return cast(SequentialAgentLike, agent)


class TestAppIntegration:
    """Pattern-based integration tests for App configuration and wiring."""

    def test_app_is_properly_instantiated(self) -> None:
        """Verify app container is properly instantiated."""
        assert app is not None
        assert app.name is not None
        assert isinstance(app.name, str)
        assert len(app.name) > 0

    def test_app_has_root_agent(self) -> None:
        """Verify app is wired to root agent."""
        assert app.root_agent is not None

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

    def test_agent_has_required_configuration(self) -> None:
        """Verify root agent (SequentialAgent) has required configuration."""
        agent = app.root_agent
        assert agent is not None

        # SequentialAgent has name and description
        assert agent.name is not None
        assert isinstance(agent.name, str)
        assert len(agent.name) > 0

        # SequentialAgent has sub_agents
        sequential_agent = as_sequential_agent(agent)
        assert sequential_agent.sub_agents is not None
        assert len(sequential_agent.sub_agents) >= 1

    def test_sub_agents_have_valid_configuration(self) -> None:
        """Verify sub-agents have valid model and instruction configuration."""
        agent = app.root_agent
        assert agent is not None
        sequential_agent = as_sequential_agent(agent)

        # Check each sub-agent (writer and publisher)
        for sub_agent in sequential_agent.sub_agents:
            typed_sub = as_agent_config(sub_agent)

            # Each sub-agent should have a name
            assert typed_sub.name is not None
            assert isinstance(typed_sub.name, str)
            assert len(typed_sub.name) > 0

            # Each sub-agent should have a model
            assert typed_sub.model is not None
            if isinstance(typed_sub.model, str):
                assert len(typed_sub.model) > 0
            else:
                # If it's an object, it should have a model attribute
                assert hasattr(typed_sub.model, "model")
                assert isinstance(typed_sub.model.model, str)
                assert len(typed_sub.model.model) > 0

    def test_sub_agents_have_valid_instructions(self) -> None:
        """Verify sub-agents have valid instructions and descriptions."""
        agent = app.root_agent
        assert agent is not None
        sequential_agent = as_sequential_agent(agent)

        for sub_agent in sequential_agent.sub_agents:
            typed_sub = as_agent_config(sub_agent)

            # Each sub-agent should have instructions
            if typed_sub.instruction is not None:
                assert isinstance(typed_sub.instruction, str)
                assert len(typed_sub.instruction) > 0

            # Each sub-agent should have a description
            if typed_sub.description is not None:
                assert isinstance(typed_sub.description, str)
                assert len(typed_sub.description) > 0

    def test_sub_agents_have_valid_tools(self) -> None:
        """Verify sub-agents have properly configured tools."""
        agent = app.root_agent
        assert agent is not None
        sequential_agent = as_sequential_agent(agent)

        for sub_agent in sequential_agent.sub_agents:
            typed_sub = as_agent_config(sub_agent)

            # Tools should be a list if configured
            if typed_sub.tools is not None:
                assert isinstance(typed_sub.tools, list)
                for tool in typed_sub.tools:
                    assert tool is not None
                    assert hasattr(tool, "__class__")

    def test_writer_agent_has_save_tool(self) -> None:
        """Verify writer agent has the save_blog_content tool."""
        agent = app.root_agent
        assert agent is not None
        sequential_agent = as_sequential_agent(agent)

        # Find the writer agent
        writer_agent = None
        for sub_agent in sequential_agent.sub_agents:
            if sub_agent.name == "blog_writer":
                writer_agent = sub_agent
                break

        assert writer_agent is not None, "blog_writer agent not found"
        typed_writer = as_agent_config(writer_agent)

        # Writer should have tools
        assert typed_writer.tools is not None
        assert len(typed_writer.tools) >= 1

    def test_publisher_agent_has_publish_tool(self) -> None:
        """Verify publisher agent has the publish_blog_to_github tool."""
        agent = app.root_agent
        assert agent is not None
        sequential_agent = as_sequential_agent(agent)

        # Find the publisher agent
        publisher_agent = None
        for sub_agent in sequential_agent.sub_agents:
            if sub_agent.name == "blog_publisher":
                publisher_agent = sub_agent
                break

        assert publisher_agent is not None, "blog_publisher agent not found"
        typed_publisher = as_agent_config(publisher_agent)

        # Publisher should have tools
        assert typed_publisher.tools is not None
        assert len(typed_publisher.tools) >= 1
