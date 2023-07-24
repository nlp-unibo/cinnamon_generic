from typing import List, Optional, Type, Set, Iterable, Any

from cinnamon_core.core.configuration import Configuration, C
from cinnamon_core.core.registry import Registration


class PipelineConfig(Configuration):

    @classmethod
    def from_keys(
            cls: Type[C],
            keys: List[Registration],
            names: List[str]
    ) -> C:
        config = cls.get_default()

        if len(keys) != len(names):
            raise AttributeError(f'Inconsistent keys and names. Got keys={keys} and names={names}')

        for key, name in zip(keys, names):
            config.add_pipeline_component(name=name,
                                          value=key)

        return config

    def add_pipeline_component(
            self,
            name: str,
            value: Optional[Any] = None,
            type_hint: Optional[Type] = None,
            description: Optional[str] = None,
            tags: Optional[Set[str]] = None,
            is_required: bool = False,
            build_type_hint: Optional[Type] = None,
            variants: Optional[Iterable] = None,
    ):
        tags = tags.union({'pipeline'}) if tags is not None else {'pipeline'}
        self.add(name=name,
                 value=value,
                 type_hint=type_hint,
                 description=description,
                 tags=tags,
                 is_required=is_required,
                 build_type_hint=build_type_hint,
                 variants=variants,
                 is_child=True)


class OrderedPipelineConfig(PipelineConfig):

    def add_pipeline_component(
            self,
            name: str,
            value: Optional[Any] = None,
            type_hint: Optional[Type] = None,
            description: Optional[str] = None,
            tags: Optional[Set[str]] = None,
            is_required: bool = False,
            build_type_hint: Optional[Type] = None,
            variants: Optional[Iterable] = None,
            order: Optional[int] = None
    ):
        super().add_pipeline_component(name=name,
                                       value=value,
                                       type_hint=type_hint,
                                       description=description,
                                       tags=tags,
                                       is_required=is_required,
                                       build_type_hint=build_type_hint,
                                       variants=variants)
        if 'ordering' not in self:
            self.add(name='ordering',
                     value=[],
                     type_hint=List[str],
                     is_required=True,
                     description="A list of Parameter names in Configuration that point to pipeline components."
                                 "This list is used to retrieve the correct order of execution of pipeline "
                                 "components: the specified ordering in this Parameter is the execution order.")

        if order is None:
            self.ordering.append(name)
        else:
            self.ordering.insert(order, name)
