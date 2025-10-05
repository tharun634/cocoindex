"""
Utilities to dump/load objects (for configs, specs).
"""

from __future__ import annotations

import datetime
import dataclasses
from enum import Enum
from typing import Any, Mapping, TypeVar, overload, get_origin

import numpy as np

from .typing import (
    AnalyzedAnyType,
    AnalyzedBasicType,
    AnalyzedDictType,
    AnalyzedListType,
    AnalyzedStructType,
    AnalyzedTypeInfo,
    AnalyzedUnionType,
    EnrichedValueType,
    FieldSchema,
    analyze_type_info,
    encode_enriched_type,
    is_namedtuple_type,
    is_pydantic_model,
    extract_ndarray_elem_dtype,
)


T = TypeVar("T")

try:
    import pydantic, pydantic_core
except ImportError:
    pass


def get_auto_default_for_type(
    type_info: AnalyzedTypeInfo,
) -> tuple[Any, bool]:
    """
    Get an auto-default value for a type annotation if it's safe to do so.

    Returns:
        A tuple of (default_value, is_supported) where:
        - default_value: The default value if auto-defaulting is supported
        - is_supported: True if auto-defaulting is supported for this type
    """
    # Case 1: Nullable types (Optional[T] or T | None)
    if type_info.nullable:
        return None, True

    # Case 2: Table types (KTable or LTable) - check if it's a list or dict type
    if isinstance(type_info.variant, AnalyzedListType):
        return [], True
    elif isinstance(type_info.variant, AnalyzedDictType):
        return {}, True

    return None, False


def dump_engine_object(v: Any) -> Any:
    """Recursively dump an object for engine. Engine side uses `Pythonized` to catch."""
    if v is None:
        return None
    elif isinstance(v, EnrichedValueType):
        return v.encode()
    elif isinstance(v, FieldSchema):
        return v.encode()
    elif isinstance(v, type) or get_origin(v) is not None:
        return encode_enriched_type(v)
    elif isinstance(v, Enum):
        return v.value
    elif isinstance(v, datetime.timedelta):
        total_secs = v.total_seconds()
        secs = int(total_secs)
        nanos = int((total_secs - secs) * 1e9)
        return {"secs": secs, "nanos": nanos}
    elif is_namedtuple_type(type(v)):
        # Handle NamedTuple objects specifically to use dict format
        field_names = list(getattr(type(v), "_fields", ()))
        result = {}
        for name in field_names:
            val = getattr(v, name)
            result[name] = dump_engine_object(val)  # Include all values, including None
        if hasattr(v, "kind") and "kind" not in result:
            result["kind"] = v.kind
        return result
    elif hasattr(v, "__dict__"):  # for dataclass-like objects
        s = {}
        for k, val in v.__dict__.items():
            if val is None:
                # Skip None values
                continue
            s[k] = dump_engine_object(val)
        if hasattr(v, "kind") and "kind" not in s:
            s["kind"] = v.kind
        return s
    elif isinstance(v, (list, tuple)):
        return [dump_engine_object(item) for item in v]
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, dict):
        return {k: dump_engine_object(v) for k, v in v.items()}
    return v


@overload
def load_engine_object(expected_type: type[T], v: Any) -> T: ...
@overload
def load_engine_object(expected_type: Any, v: Any) -> Any: ...
def load_engine_object(expected_type: Any, v: Any) -> Any:
    """Recursively load an object that was produced by dump_engine_object().

    Args:
        expected_type: The Python type annotation to reconstruct to.
        v: The engine-facing Pythonized object (e.g., dict/list/primitive) to convert.

    Returns:
        A Python object matching the expected_type where possible.
    """
    # Fast path
    if v is None:
        return None

    type_info = analyze_type_info(expected_type)
    variant = type_info.variant

    if type_info.core_type is EnrichedValueType:
        return EnrichedValueType.decode(v)
    if type_info.core_type is FieldSchema:
        return FieldSchema.decode(v)

    # Any or unknown → return as-is
    if isinstance(variant, AnalyzedAnyType) or type_info.base_type is Any:
        return v

    # Enum handling
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return expected_type(v)

    # TimeDelta special form {secs, nanos}
    if isinstance(variant, AnalyzedBasicType) and variant.kind == "TimeDelta":
        if isinstance(v, Mapping) and "secs" in v and "nanos" in v:
            secs = int(v["secs"])  # type: ignore[index]
            nanos = int(v["nanos"])  # type: ignore[index]
            return datetime.timedelta(seconds=secs, microseconds=nanos / 1_000)
        return v

    # List, NDArray (Vector-ish), or general sequences
    if isinstance(variant, AnalyzedListType):
        elem_type = variant.elem_type if variant.elem_type else Any
        if type_info.base_type is np.ndarray:
            # Reconstruct NDArray with appropriate dtype if available
            try:
                dtype = extract_ndarray_elem_dtype(type_info.core_type)
            except (TypeError, ValueError, AttributeError):
                dtype = None
            return np.array(v, dtype=dtype)
        # Regular Python list
        return [load_engine_object(elem_type, item) for item in v]

    # Dict / Mapping
    if isinstance(variant, AnalyzedDictType):
        key_t = variant.key_type
        val_t = variant.value_type
        return {
            load_engine_object(key_t, k): load_engine_object(val_t, val)
            for k, val in v.items()
        }

    # Structs (dataclass, NamedTuple, or Pydantic)
    if isinstance(variant, AnalyzedStructType):
        struct_type = variant.struct_type
        init_kwargs: dict[str, Any] = {}
        missing_fields: list[tuple[str, Any]] = []
        if dataclasses.is_dataclass(struct_type):
            if not isinstance(v, Mapping):
                raise ValueError(f"Expected dict for dataclass, got {type(v)}")

            for dc_field in dataclasses.fields(struct_type):
                if dc_field.name in v:
                    init_kwargs[dc_field.name] = load_engine_object(
                        dc_field.type, v[dc_field.name]
                    )
                else:
                    if (
                        dc_field.default is dataclasses.MISSING
                        and dc_field.default_factory is dataclasses.MISSING
                    ):
                        missing_fields.append((dc_field.name, dc_field.type))

        elif is_namedtuple_type(struct_type):
            if not isinstance(v, Mapping):
                raise ValueError(f"Expected dict for NamedTuple, got {type(v)}")
            # Dict format (from dump/load functions)
            annotations = getattr(struct_type, "__annotations__", {})
            field_names = list(getattr(struct_type, "_fields", ()))
            field_defaults = getattr(struct_type, "_field_defaults", {})

            for name in field_names:
                f_type = annotations.get(name, Any)
                if name in v:
                    init_kwargs[name] = load_engine_object(f_type, v[name])
                elif name not in field_defaults:
                    missing_fields.append((name, f_type))

        elif is_pydantic_model(struct_type):
            if not isinstance(v, Mapping):
                raise ValueError(f"Expected dict for Pydantic model, got {type(v)}")

            model_fields: dict[str, pydantic.fields.FieldInfo]
            if hasattr(struct_type, "model_fields"):
                model_fields = struct_type.model_fields  # type: ignore[attr-defined]
            else:
                model_fields = {}

            for name, pyd_field in model_fields.items():
                if name in v:
                    init_kwargs[name] = load_engine_object(
                        pyd_field.annotation, v[name]
                    )
                elif (
                    getattr(pyd_field, "default", pydantic_core.PydanticUndefined)
                    is pydantic_core.PydanticUndefined
                    and getattr(pyd_field, "default_factory") is None
                ):
                    missing_fields.append((name, pyd_field.annotation))
        else:
            assert False, "Unsupported struct type"

        for name, f_type in missing_fields:
            type_info = analyze_type_info(f_type)
            auto_default, is_supported = get_auto_default_for_type(type_info)
            if is_supported:
                init_kwargs[name] = auto_default
        return struct_type(**init_kwargs)

    # Union with discriminator support via "kind"
    if isinstance(variant, AnalyzedUnionType):
        if isinstance(v, Mapping) and "kind" in v:
            discriminator = v["kind"]
            for typ in variant.variant_types:
                t_info = analyze_type_info(typ)
                if isinstance(t_info.variant, AnalyzedStructType):
                    t_struct = t_info.variant.struct_type
                    candidate_kind = getattr(t_struct, "kind", None)
                    if candidate_kind == discriminator:
                        # Remove discriminator for constructor
                        v_wo_kind = dict(v)
                        v_wo_kind.pop("kind", None)
                        return load_engine_object(t_struct, v_wo_kind)
        # Fallback: try each variant until one succeeds
        for typ in variant.variant_types:
            try:
                return load_engine_object(typ, v)
            except (TypeError, ValueError):
                continue
        return v

    # Basic types and everything else: handle numpy scalars and passthrough
    if isinstance(v, np.ndarray) and type_info.base_type is list:
        return v.tolist()
    if isinstance(v, (list, tuple)) and type_info.base_type not in (list, tuple):
        # If a non-sequence basic type expected, attempt direct cast
        try:
            return type_info.core_type(v)
        except (TypeError, ValueError):
            return v
    return v
