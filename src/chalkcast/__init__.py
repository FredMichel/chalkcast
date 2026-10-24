"""Chalkcast — describe a concept, get a verified animated explainer.

The MCP server lives in ``chalkcast.server`` and depends on ``mcp``. It is imported
lazily (not at package import) so the pure authoring/verification modules work
anywhere, including where the render/MCP extras are not installed.
"""

__version__ = "0.1.0"


def get_mcp():
    """Return the FastMCP server instance (imports ``mcp`` on demand)."""
    from .server import mcp
    return mcp
