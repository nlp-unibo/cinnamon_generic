from cinnamon_core.core.configuration import Configuration
from cinnamon_core.core.registry import register, Registry

from cinnamon_generic.components.routine_processor import AverageProcessor, FoldProcessor


@register
def register_routine_processor():
    Registry.add_and_bind(config_class=Configuration,
                          component_class=AverageProcessor,
                          name='routine_processor',
                          tags={'average'},
                          namespace='generic')
    Registry.add_and_bind(config_class=Configuration,
                          component_class=FoldProcessor,
                          name='routine_processor',
                          tags={'average', 'fold'},
                          namespace='generic')
