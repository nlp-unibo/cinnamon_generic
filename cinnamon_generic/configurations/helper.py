from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import Registry, register

from cinnamon_generic.components.helper import Helper


@register
def register_helper_configurations():
    Registry.add_and_bind(config_class=Configuration,
                          component_class=Helper,
                          name='helper',
                          namespace='generic',
                          is_default=True)
