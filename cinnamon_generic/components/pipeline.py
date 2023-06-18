from typing import List

from cinnamon_core.core.component import Component
from cinnamon_generic.configurations.pipeline import PipelineConfig


class Pipeline(Component):

    @classmethod
    def from_components(
            cls,
            components: List[Component]
    ):
        pipeline = cls(config=PipelineConfig())

        for idx, component in enumerate(components):
            pipeline.config.add(name=f'component_{idx}',
                                value=component,
                                tags={'pipeline'})

        return pipeline

    def get_pipeline(
            self
    ) -> List[Component]:
        components = self.config.search_by_tag(tags='pipeline',
                                               exact_match=False)
        return list(components.values())


class OrderedPipeline(Pipeline):

    def get_pipeline(
            self
    ) -> List[Component]:
        components = self.config.search_by_tag(tags='pipeline',
                                               exact_match=False)
        components = [components[key] for key in self.ordering]
        return components
