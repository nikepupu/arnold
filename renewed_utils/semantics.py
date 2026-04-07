from pxr import Sdf, Usd


def add_update_semantics(prim: Usd.Prim, semantic_label: str, type_label: str = "class", suffix: str = "") -> None:
    """Drop-in replacement for isaacsim.core.utils.semantics.add_update_semantics
    that uses raw USD attribute APIs instead of the deprecated Semantics.SemanticsAPI,
    avoiding the C++-level deprecation warning that cannot be suppressed via logging config.
    """
    instance_name = "Semantics" + suffix
    type_attr_name = f"semantic:{instance_name}:params:semanticType"
    data_attr_name = f"semantic:{instance_name}:params:semanticData"

    type_attr = prim.GetAttribute(type_attr_name)
    if not type_attr or not type_attr.IsValid():
        type_attr = prim.CreateAttribute(type_attr_name, Sdf.ValueTypeNames.String, custom=True)
    data_attr = prim.GetAttribute(data_attr_name)
    if not data_attr or not data_attr.IsValid():
        data_attr = prim.CreateAttribute(data_attr_name, Sdf.ValueTypeNames.String, custom=True)

    if type_label is not None:
        type_attr.Set(type_label)
    if semantic_label is not None:
        data_attr.Set(semantic_label)
