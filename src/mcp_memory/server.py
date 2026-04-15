"""MCP Memory server — modular entry point."""

import logging
import sys

from fastmcp import FastMCP

from mcp_memory.storage import MemoryStore

logger = logging.getLogger(__name__)

# ============================================================
# Server + Store initialization
# ============================================================
mcp = FastMCP("memory")

# Instantiate store — default path
store = MemoryStore()
store.init_db()

# ============================================================
# Tool registration — import from modules and register with FastMCP
# ============================================================

# Core CRUD tools
from mcp_memory.tools.core import (
    create_entities,
    create_relations,
    add_observations,
    delete_entities,
    delete_observations,
    delete_relations,
)

# Search tools
from mcp_memory.tools.search import (
    search_nodes,
    open_nodes,
    search_semantic,
)

# Entity management tools
from mcp_memory.tools.entity_mgmt import (
    analyze_entity_split,
    propose_entity_split_tool,
    execute_entity_split_tool,
    find_split_candidates,
    find_duplicate_observations,
    consolidation_report,
)

# Reflection tools
from mcp_memory.tools.reflections import (
    add_reflection,
    search_reflections,
)

# Relation and migration tools
from mcp_memory.tools.relations import (
    migrate,
    end_relation,
)

# Register all tools with FastMCP
mcp.tool(create_entities)
mcp.tool(create_relations)
mcp.tool(add_observations)
mcp.tool(delete_entities)
mcp.tool(delete_observations)
mcp.tool(delete_relations)
mcp.tool(search_nodes)
mcp.tool(open_nodes)
mcp.tool(search_semantic)
mcp.tool(migrate)
mcp.tool(analyze_entity_split)
mcp.tool(propose_entity_split_tool)
mcp.tool(execute_entity_split_tool)
mcp.tool(find_split_candidates)
mcp.tool(find_duplicate_observations)
mcp.tool(consolidation_report)
mcp.tool(add_reflection)
mcp.tool(search_reflections)
mcp.tool(end_relation)


# ============================================================
# Main entry point
# ============================================================
def main() -> None:
    """Start the MCP server (stdio transport)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,  # Log to stderr, not stdout (MCP uses stdout)
    )
    logger.info("Starting MCP Memory v2 server...")
    mcp.run()
