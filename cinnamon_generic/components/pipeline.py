from cinnamon_core.core.component import Component
from typing import List


class Pipeline(Component):

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
