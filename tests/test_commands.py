from pathlib import Path

from cinnamon_core.core.registry import Registry
from cinnamon_generic.api.commands import setup_registry


def test_setup():
    directory = Path(__file__).parent.parent.resolve()
    setup_registry(directory=directory,
                   registrations_to_file=False)
    assert len(Registry.REGISTRY) == 8
