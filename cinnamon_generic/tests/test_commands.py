from pathlib import Path

from cinnamon_generic.api.commands import setup_registry
from cinnamon_core.core.registry import Registry


def test_setup():
    directory = Path(__file__).parent.parent.resolve()
    setup_registry(directory=directory,
                   registrations_to_file=False)
    assert len(Registry.REGISTRY) == 6
