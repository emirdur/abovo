import sys
from unittest.mock import MagicMock

# Create a mock class for the _abovo module
class MockModule(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Mock the _abovo module
sys.modules["_abovo"] = MockModule()