"""Bridge module — re-exports the full SchemaAgent with analyze() from agents.py.

The sqlagent/agents/ package has a lightweight SchemaAgent (agents/schema.py)
used by the orchestrator for table introspection. The FULL SchemaAgent with
the 5-stage LLM analysis pipeline lives in agents.py (standalone file).

Python resolves `from sqlagent.agents import SchemaAgent` to the package version.
This bridge provides `from sqlagent.schema_agent import FullSchemaAgent` to access
the analysis version without naming conflicts.
"""

# agents.py is importable as sqlagent.agents only when the package doesn't shadow it.
# Since sqlagent/agents/ (directory) exists, we re-export by reading the module directly.

import sys
import types

# The standalone agents.py is already loaded as part of various imports.
# We just need to find the SchemaAgent class from it.
# Since direct import won't work (package shadows file), we use a function-level import
# that loads the specific class we need.

def _get_full_schema_agent():
    """Load the full SchemaAgent from the standalone agents.py file."""
    import importlib
    import os

    # Load the standalone agents.py as a separate module name
    agents_path = os.path.join(os.path.dirname(__file__), "agents.py")
    spec = importlib.util.spec_from_file_location("sqlagent._agents_standalone", agents_path)
    mod = types.ModuleType("sqlagent._agents_standalone")
    mod.__file__ = agents_path
    mod.__spec__ = spec
    # Register in sys.modules so dataclass decorator can find __module__
    sys.modules["sqlagent._agents_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod.SchemaAgent


# Eager load so `from sqlagent.schema_agent import FullSchemaAgent` works
FullSchemaAgent = _get_full_schema_agent()
