import carb
from isaacsim.core.utils.semantics import add_update_semantics as _orig_add_update_semantics


def add_update_semantics(prim, semantic_label, type_label="class"):
    """Wrapper that silences the deprecated-SemanticsAPI warning."""
    prev = carb.settings.get_settings().get("/log/outputStreamLevel")
    carb.settings.get_settings().set("/log/outputStreamLevel", carb.logging.LEVEL_ERROR)
    try:
        return _orig_add_update_semantics(prim, semantic_label, type_label)
    finally:
        carb.settings.get_settings().set(
            "/log/outputStreamLevel",
            prev if prev is not None else carb.logging.LEVEL_WARN,
        )
