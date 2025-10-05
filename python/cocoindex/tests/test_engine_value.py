import datetime
import inspect
import uuid
from dataclasses import dataclass, make_dataclass
from typing import Annotated, Any, Callable, Literal, NamedTuple, Type

import numpy as np
import pytest
from numpy.typing import NDArray

# Optional Pydantic support for testing
try:
    from pydantic import BaseModel, Field

    PYDANTIC_AVAILABLE = True
except ImportError:
    BaseModel = None  # type: ignore[misc,assignment]
    Field = None  # type: ignore[misc,assignment]
    PYDANTIC_AVAILABLE = False

import cocoindex
from cocoindex.engine_value import (
    make_engine_value_encoder,
    make_engine_value_decoder,
)
from cocoindex.typing import (
    Float32,
    Float64,
    TypeKind,
    Vector,
    analyze_type_info,
    encode_enriched_type,
    decode_engine_value_type,
)


@dataclass
class Order:
    order_id: str
    name: str
    price: float
    extra_field: str = "default_extra"


@dataclass
class Tag:
    name: str


@dataclass
class Basket:
    items: list[str]


@dataclass
class Customer:
    name: str
    order: Order
    tags: list[Tag] | None = None


@dataclass
class NestedStruct:
    customer: Customer
    orders: list[Order]
    count: int = 0


class OrderNamedTuple(NamedTuple):
    order_id: str
    name: str
    price: float
    extra_field: str = "default_extra"


class CustomerNamedTuple(NamedTuple):
    name: str
    order: OrderNamedTuple
    tags: list[Tag] | None = None


# Pydantic model definitions (if available)
if PYDANTIC_AVAILABLE:

    class OrderPydantic(BaseModel):
        order_id: str
        name: str
        price: float
        extra_field: str = "default_extra"

    class TagPydantic(BaseModel):
        name: str

    class CustomerPydantic(BaseModel):
        name: str
        order: OrderPydantic
        tags: list[TagPydantic] | None = None

    class NestedStructPydantic(BaseModel):
        customer: CustomerPydantic
        orders: list[OrderPydantic]
        count: int = 0


def encode_engine_value(value: Any, type_hint: Type[Any] | str) -> Any:
    """
    Encode a Python value to an engine value.
    """
    encoder = make_engine_value_encoder(analyze_type_info(type_hint))
    return encoder(value)


def build_engine_value_decoder(
    engine_type_in_py: Any, python_type: Any | None = None
) -> Callable[[Any], Any]:
    """
    Helper to build a converter for the given engine-side type (as represented in Python).
    If python_type is not specified, uses engine_type_in_py as the target.
    """
    engine_type = encode_enriched_type(engine_type_in_py)["type"]
    return make_engine_value_decoder(
        [],
        decode_engine_value_type(engine_type),
        analyze_type_info(python_type or engine_type_in_py),
    )


def validate_full_roundtrip_to(
    value: Any,
    value_type: Any,
    *decoded_values: tuple[Any, Any],
) -> None:
    """
    Validate the given value becomes specific values after encoding, sending to engine (using output_type), receiving back and decoding (using input_type).

    `decoded_values` is a tuple of (value, type) pairs.
    """
    from cocoindex import _engine  # type: ignore

    def eq(a: Any, b: Any) -> bool:
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        return type(a) is type(b) and not not (a == b)

    encoded_value = encode_engine_value(value, value_type)
    value_type = value_type or type(value)
    encoded_output_type = encode_enriched_type(value_type)["type"]
    value_from_engine = _engine.testutil.seder_roundtrip(
        encoded_value, encoded_output_type
    )

    for other_value, other_type in decoded_values:
        decoder = make_engine_value_decoder(
            [],
            decode_engine_value_type(encoded_output_type),
            analyze_type_info(other_type),
        )
        other_decoded_value = decoder(value_from_engine)
        assert eq(other_decoded_value, other_value), (
            f"Expected {other_value} but got {other_decoded_value} for {other_type}"
        )


def validate_full_roundtrip(
    value: Any,
    value_type: Any,
    *other_decoded_values: tuple[Any, Any],
) -> None:
    """
    Validate the given value doesn't change after encoding, sending to engine (using output_type), receiving back and decoding (using input_type).

    `other_decoded_values` is a tuple of (value, type) pairs.
    If provided, also validate the value can be decoded to the other types.
    """
    validate_full_roundtrip_to(
        value, value_type, (value, value_type), *other_decoded_values
    )


def test_encode_engine_value_basic_types() -> None:
    assert encode_engine_value(123, int) == 123
    assert encode_engine_value(3.14, float) == 3.14
    assert encode_engine_value("hello", str) == "hello"
    assert encode_engine_value(True, bool) is True


def test_encode_engine_value_uuid() -> None:
    u = uuid.uuid4()
    assert encode_engine_value(u, uuid.UUID) == u


def test_encode_engine_value_date_time_types() -> None:
    d = datetime.date(2024, 1, 1)
    assert encode_engine_value(d, datetime.date) == d
    t = datetime.time(12, 30)
    assert encode_engine_value(t, datetime.time) == t
    dt = datetime.datetime(2024, 1, 1, 12, 30)
    assert encode_engine_value(dt, datetime.datetime) == dt


def test_encode_scalar_numpy_values() -> None:
    """Test encoding scalar NumPy values to engine-compatible values."""
    test_cases = [
        (np.int64(42), 42),
        (np.float32(3.14), pytest.approx(3.14)),
        (np.float64(2.718), pytest.approx(2.718)),
    ]
    for np_value, expected in test_cases:
        encoded = encode_engine_value(np_value, type(np_value))
        assert encoded == expected
        assert isinstance(encoded, (int, float))


def test_encode_engine_value_struct() -> None:
    order = Order(order_id="O123", name="mixed nuts", price=25.0)
    assert encode_engine_value(order, Order) == [
        "O123",
        "mixed nuts",
        25.0,
        "default_extra",
    ]

    order_nt = OrderNamedTuple(order_id="O123", name="mixed nuts", price=25.0)
    assert encode_engine_value(order_nt, OrderNamedTuple) == [
        "O123",
        "mixed nuts",
        25.0,
        "default_extra",
    ]


def test_encode_engine_value_list_of_structs() -> None:
    orders = [Order("O1", "item1", 10.0), Order("O2", "item2", 20.0)]
    assert encode_engine_value(orders, list[Order]) == [
        ["O1", "item1", 10.0, "default_extra"],
        ["O2", "item2", 20.0, "default_extra"],
    ]

    orders_nt = [
        OrderNamedTuple("O1", "item1", 10.0),
        OrderNamedTuple("O2", "item2", 20.0),
    ]
    assert encode_engine_value(orders_nt, list[OrderNamedTuple]) == [
        ["O1", "item1", 10.0, "default_extra"],
        ["O2", "item2", 20.0, "default_extra"],
    ]


def test_encode_engine_value_struct_with_list() -> None:
    basket = Basket(items=["apple", "banana"])
    assert encode_engine_value(basket, Basket) == [["apple", "banana"]]


def test_encode_engine_value_nested_struct() -> None:
    customer = Customer(name="Alice", order=Order("O1", "item1", 10.0))
    assert encode_engine_value(customer, Customer) == [
        "Alice",
        ["O1", "item1", 10.0, "default_extra"],
        None,
    ]

    customer_nt = CustomerNamedTuple(
        name="Alice", order=OrderNamedTuple("O1", "item1", 10.0)
    )
    assert encode_engine_value(customer_nt, CustomerNamedTuple) == [
        "Alice",
        ["O1", "item1", 10.0, "default_extra"],
        None,
    ]


def test_encode_engine_value_empty_list() -> None:
    assert encode_engine_value([], list) == []
    assert encode_engine_value([[]], list[list[Any]]) == [[]]


def test_encode_engine_value_tuple() -> None:
    assert encode_engine_value((), Any) == []
    assert encode_engine_value((1, 2, 3), Any) == [1, 2, 3]
    assert encode_engine_value(((1, 2), (3, 4)), Any) == [[1, 2], [3, 4]]
    assert encode_engine_value(([],), Any) == [[]]
    assert encode_engine_value(((),), Any) == [[]]


def test_encode_engine_value_none() -> None:
    assert encode_engine_value(None, Any) is None


def test_roundtrip_basic_types() -> None:
    validate_full_roundtrip(
        b"hello world",
        bytes,
        (b"hello world", inspect.Parameter.empty),
        (b"hello world", Any),
    )
    validate_full_roundtrip(b"\x00\x01\x02\xff\xfe", bytes)
    validate_full_roundtrip("hello", str, ("hello", Any))
    validate_full_roundtrip(True, bool, (True, Any))
    validate_full_roundtrip(False, bool, (False, Any))
    validate_full_roundtrip(
        42, cocoindex.Int64, (42, int), (np.int64(42), np.int64), (42, Any)
    )
    validate_full_roundtrip(42, int, (42, cocoindex.Int64))
    validate_full_roundtrip(np.int64(42), np.int64, (42, cocoindex.Int64))

    validate_full_roundtrip(
        3.25, Float64, (3.25, float), (np.float64(3.25), np.float64), (3.25, Any)
    )
    validate_full_roundtrip(3.25, float, (3.25, Float64))
    validate_full_roundtrip(np.float64(3.25), np.float64, (3.25, Float64))

    validate_full_roundtrip(
        3.25,
        Float32,
        (3.25, float),
        (np.float32(3.25), np.float32),
        (np.float64(3.25), np.float64),
        (3.25, Float64),
        (3.25, Any),
    )
    validate_full_roundtrip(np.float32(3.25), np.float32, (3.25, Float32))


def test_roundtrip_uuid() -> None:
    uuid_value = uuid.uuid4()
    validate_full_roundtrip(uuid_value, uuid.UUID, (uuid_value, Any))


def test_roundtrip_range() -> None:
    r1 = (0, 100)
    validate_full_roundtrip(r1, cocoindex.Range, (r1, Any))
    r2 = (50, 50)
    validate_full_roundtrip(r2, cocoindex.Range, (r2, Any))
    r3 = (0, 1_000_000_000)
    validate_full_roundtrip(r3, cocoindex.Range, (r3, Any))


def test_roundtrip_time() -> None:
    t1 = datetime.time(10, 30, 50, 123456)
    validate_full_roundtrip(t1, datetime.time, (t1, Any))
    t2 = datetime.time(23, 59, 59)
    validate_full_roundtrip(t2, datetime.time, (t2, Any))
    t3 = datetime.time(0, 0, 0)
    validate_full_roundtrip(t3, datetime.time, (t3, Any))

    validate_full_roundtrip(
        datetime.date(2025, 1, 1), datetime.date, (datetime.date(2025, 1, 1), Any)
    )

    validate_full_roundtrip(
        datetime.datetime(2025, 1, 2, 3, 4, 5, 123456),
        cocoindex.LocalDateTime,
        (datetime.datetime(2025, 1, 2, 3, 4, 5, 123456), datetime.datetime),
    )

    tz = datetime.timezone(datetime.timedelta(hours=5))
    validate_full_roundtrip(
        datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, tz),
        cocoindex.OffsetDateTime,
        (
            datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, tz),
            datetime.datetime,
        ),
    )
    validate_full_roundtrip(
        datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, tz),
        datetime.datetime,
        (datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, tz), cocoindex.OffsetDateTime),
    )
    validate_full_roundtrip_to(
        datetime.datetime(2025, 1, 2, 3, 4, 5, 123456),
        cocoindex.OffsetDateTime,
        (
            datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, datetime.UTC),
            datetime.datetime,
        ),
    )
    validate_full_roundtrip_to(
        datetime.datetime(2025, 1, 2, 3, 4, 5, 123456),
        datetime.datetime,
        (
            datetime.datetime(2025, 1, 2, 3, 4, 5, 123456, datetime.UTC),
            cocoindex.OffsetDateTime,
        ),
    )


def test_roundtrip_timedelta() -> None:
    td1 = datetime.timedelta(
        days=5, seconds=10, microseconds=123, milliseconds=456, minutes=30, hours=2
    )
    validate_full_roundtrip(td1, datetime.timedelta, (td1, Any))
    td2 = datetime.timedelta(days=-5, hours=-2)
    validate_full_roundtrip(td2, datetime.timedelta, (td2, Any))
    td3 = datetime.timedelta(0)
    validate_full_roundtrip(td3, datetime.timedelta, (td3, Any))


def test_roundtrip_json() -> None:
    simple_dict = {"key": "value", "number": 123, "bool": True, "float": 1.23}
    validate_full_roundtrip(simple_dict, cocoindex.Json)

    simple_list = [1, "string", False, None, 4.56]
    validate_full_roundtrip(simple_list, cocoindex.Json)

    nested_structure = {
        "name": "Test Json",
        "version": 1.0,
        "items": [
            {"id": 1, "value": "item1"},
            {"id": 2, "value": None, "props": {"active": True}},
        ],
        "metadata": None,
    }
    validate_full_roundtrip(nested_structure, cocoindex.Json)

    validate_full_roundtrip({}, cocoindex.Json)
    validate_full_roundtrip([], cocoindex.Json)


def test_decode_scalar_numpy_values() -> None:
    test_cases = [
        (decode_engine_value_type({"kind": "Int64"}), np.int64, 42, np.int64(42)),
        (
            decode_engine_value_type({"kind": "Float32"}),
            np.float32,
            3.14,
            np.float32(3.14),
        ),
        (
            decode_engine_value_type({"kind": "Float64"}),
            np.float64,
            2.718,
            np.float64(2.718),
        ),
    ]
    for src_type, dst_type, input_value, expected in test_cases:
        decoder = make_engine_value_decoder(
            ["field"], src_type, analyze_type_info(dst_type)
        )
        result = decoder(input_value)
        assert isinstance(result, dst_type)
        assert result == expected


def test_non_ndarray_vector_decoding() -> None:
    # Test list[np.float64]
    src_type = decode_engine_value_type(
        {
            "kind": "Vector",
            "element_type": {"kind": "Float64"},
            "dimension": None,
        }
    )
    dst_type_float = list[np.float64]
    decoder = make_engine_value_decoder(
        ["field"], src_type, analyze_type_info(dst_type_float)
    )
    input_numbers = [1.0, 2.0, 3.0]
    result = decoder(input_numbers)
    assert isinstance(result, list)
    assert all(isinstance(x, np.float64) for x in result)
    assert result == [np.float64(1.0), np.float64(2.0), np.float64(3.0)]

    # Test list[Uuid]
    src_type = decode_engine_value_type(
        {"kind": "Vector", "element_type": {"kind": "Uuid"}, "dimension": None}
    )
    dst_type_uuid = list[uuid.UUID]
    decoder = make_engine_value_decoder(
        ["field"], src_type, analyze_type_info(dst_type_uuid)
    )
    uuid1 = uuid.uuid4()
    uuid2 = uuid.uuid4()
    input_uuids = [uuid1, uuid2]
    result = decoder(input_uuids)
    assert isinstance(result, list)
    assert all(isinstance(x, uuid.UUID) for x in result)
    assert result == [uuid1, uuid2]


def test_roundtrip_struct() -> None:
    validate_full_roundtrip(
        Order("O123", "mixed nuts", 25.0, "default_extra"),
        Order,
    )
    validate_full_roundtrip(
        OrderNamedTuple("O123", "mixed nuts", 25.0, "default_extra"),
        OrderNamedTuple,
    )


def test_make_engine_value_decoder_list_of_struct() -> None:
    # List of structs (dataclass)
    engine_val = [
        ["O1", "item1", 10.0, "default_extra"],
        ["O2", "item2", 20.0, "default_extra"],
    ]
    decoder = build_engine_value_decoder(list[Order])
    assert decoder(engine_val) == [
        Order("O1", "item1", 10.0, "default_extra"),
        Order("O2", "item2", 20.0, "default_extra"),
    ]

    # List of structs (NamedTuple)
    decoder = build_engine_value_decoder(list[OrderNamedTuple])
    assert decoder(engine_val) == [
        OrderNamedTuple("O1", "item1", 10.0, "default_extra"),
        OrderNamedTuple("O2", "item2", 20.0, "default_extra"),
    ]


def test_make_engine_value_decoder_struct_of_list() -> None:
    # Struct with list field
    engine_val = [
        "Alice",
        ["O1", "item1", 10.0, "default_extra"],
        [["vip"], ["premium"]],
    ]
    decoder = build_engine_value_decoder(Customer)
    assert decoder(engine_val) == Customer(
        "Alice",
        Order("O1", "item1", 10.0, "default_extra"),
        [Tag("vip"), Tag("premium")],
    )

    # NamedTuple with list field
    decoder = build_engine_value_decoder(CustomerNamedTuple)
    assert decoder(engine_val) == CustomerNamedTuple(
        "Alice",
        OrderNamedTuple("O1", "item1", 10.0, "default_extra"),
        [Tag("vip"), Tag("premium")],
    )


def test_make_engine_value_decoder_struct_of_struct() -> None:
    # Struct with struct field
    engine_val = [
        ["Alice", ["O1", "item1", 10.0, "default_extra"], [["vip"]]],
        [
            ["O1", "item1", 10.0, "default_extra"],
            ["O2", "item2", 20.0, "default_extra"],
        ],
        2,
    ]
    decoder = build_engine_value_decoder(NestedStruct)
    assert decoder(engine_val) == NestedStruct(
        Customer("Alice", Order("O1", "item1", 10.0, "default_extra"), [Tag("vip")]),
        [
            Order("O1", "item1", 10.0, "default_extra"),
            Order("O2", "item2", 20.0, "default_extra"),
        ],
        2,
    )


def make_engine_order(fields: list[tuple[str, type]]) -> type:
    return make_dataclass("EngineOrder", fields)


def make_python_order(
    fields: list[tuple[str, type]], defaults: dict[str, Any] | None = None
) -> type:
    if defaults is None:
        defaults = {}
    # Move all fields with defaults to the end (Python dataclass requirement)
    non_default_fields = [(n, t) for n, t in fields if n not in defaults]
    default_fields = [(n, t) for n, t in fields if n in defaults]
    ordered_fields = non_default_fields + default_fields
    # Prepare the namespace for defaults (only for fields at the end)
    namespace = {k: defaults[k] for k, _ in default_fields}
    return make_dataclass("PythonOrder", ordered_fields, namespace=namespace)


@pytest.mark.parametrize(
    "engine_fields, python_fields, python_defaults, engine_val, expected_python_val",
    [
        # Extra field in Python (middle)
        (
            [("id", str), ("name", str)],
            [("id", str), ("price", float), ("name", str)],
            {"price": 0.0},
            ["O123", "mixed nuts"],
            ("O123", 0.0, "mixed nuts"),
        ),
        # Missing field in Python (middle)
        (
            [("id", str), ("price", float), ("name", str)],
            [("id", str), ("name", str)],
            {},
            ["O123", 25.0, "mixed nuts"],
            ("O123", "mixed nuts"),
        ),
        # Extra field in Python (start)
        (
            [("name", str), ("price", float)],
            [("extra", str), ("name", str), ("price", float)],
            {"extra": "default"},
            ["mixed nuts", 25.0],
            ("default", "mixed nuts", 25.0),
        ),
        # Missing field in Python (start)
        (
            [("extra", str), ("name", str), ("price", float)],
            [("name", str), ("price", float)],
            {},
            ["unexpected", "mixed nuts", 25.0],
            ("mixed nuts", 25.0),
        ),
        # Field order difference (should map by name)
        (
            [("id", str), ("name", str), ("price", float)],
            [("name", str), ("id", str), ("price", float), ("extra", str)],
            {"extra": "default"},
            ["O123", "mixed nuts", 25.0],
            ("mixed nuts", "O123", 25.0, "default"),
        ),
        # Extra field (Python has extra field with default)
        (
            [("id", str), ("name", str)],
            [("id", str), ("name", str), ("price", float)],
            {"price": 0.0},
            ["O123", "mixed nuts"],
            ("O123", "mixed nuts", 0.0),
        ),
        # Missing field (Engine has extra field)
        (
            [("id", str), ("name", str), ("price", float)],
            [("id", str), ("name", str)],
            {},
            ["O123", "mixed nuts", 25.0],
            ("O123", "mixed nuts"),
        ),
    ],
)
def test_field_position_cases(
    engine_fields: list[tuple[str, type]],
    python_fields: list[tuple[str, type]],
    python_defaults: dict[str, Any],
    engine_val: list[Any],
    expected_python_val: tuple[Any, ...],
) -> None:
    EngineOrder = make_engine_order(engine_fields)
    PythonOrder = make_python_order(python_fields, python_defaults)
    decoder = build_engine_value_decoder(EngineOrder, PythonOrder)
    # Map field names to expected values
    expected_dict = dict(zip([f[0] for f in python_fields], expected_python_val))
    # Instantiate using keyword arguments (order doesn't matter)
    assert decoder(engine_val) == PythonOrder(**expected_dict)


def test_roundtrip_union_simple() -> None:
    t = int | str | float
    value = 10.4
    validate_full_roundtrip(value, t)


def test_roundtrip_union_with_active_uuid() -> None:
    t = str | uuid.UUID | int
    value = uuid.uuid4()
    validate_full_roundtrip(value, t)


def test_roundtrip_union_with_inactive_uuid() -> None:
    t = str | uuid.UUID | int
    value = "5a9f8f6a-318f-4f1f-929d-566d7444a62d"  # it's a string
    validate_full_roundtrip(value, t)


def test_roundtrip_union_offset_datetime() -> None:
    t = str | uuid.UUID | float | int | datetime.datetime
    value = datetime.datetime.now(datetime.UTC)
    validate_full_roundtrip(value, t)


def test_roundtrip_union_date() -> None:
    t = str | uuid.UUID | float | int | datetime.date
    value = datetime.date.today()
    validate_full_roundtrip(value, t)


def test_roundtrip_union_time() -> None:
    t = str | uuid.UUID | float | int | datetime.time
    value = datetime.time()
    validate_full_roundtrip(value, t)


def test_roundtrip_union_timedelta() -> None:
    t = str | uuid.UUID | float | int | datetime.timedelta
    value = datetime.timedelta(hours=39, minutes=10, seconds=1)
    validate_full_roundtrip(value, t)


def test_roundtrip_vector_of_union() -> None:
    t = list[str | int]
    value = ["a", 1]
    validate_full_roundtrip(value, t)


def test_roundtrip_union_with_vector() -> None:
    t = NDArray[np.float32] | str
    value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    validate_full_roundtrip(value, t, ([1.0, 2.0, 3.0], list[float] | str))


def test_roundtrip_union_with_misc_types() -> None:
    t_bytes_union = int | bytes | str
    validate_full_roundtrip(b"test_bytes", t_bytes_union)
    validate_full_roundtrip(123, t_bytes_union)

    t_range_union = cocoindex.Range | str | bool
    validate_full_roundtrip((100, 200), t_range_union)
    validate_full_roundtrip("test_string", t_range_union)

    t_json_union = cocoindex.Json | int | bytes
    json_dict = {"a": 1, "b": [2, 3]}
    validate_full_roundtrip(json_dict, t_json_union)
    validate_full_roundtrip(b"another_byte_string", t_json_union)


def test_roundtrip_ltable() -> None:
    t = list[Order]
    value = [Order("O1", "item1", 10.0), Order("O2", "item2", 20.0)]
    validate_full_roundtrip(value, t)

    t_nt = list[OrderNamedTuple]
    value_nt = [
        OrderNamedTuple("O1", "item1", 10.0),
        OrderNamedTuple("O2", "item2", 20.0),
    ]
    validate_full_roundtrip(value_nt, t_nt)


def test_roundtrip_ktable_various_key_types() -> None:
    @dataclass
    class SimpleValue:
        data: str

    t_bytes_key = dict[bytes, SimpleValue]
    value_bytes_key = {b"key1": SimpleValue("val1"), b"key2": SimpleValue("val2")}
    validate_full_roundtrip(value_bytes_key, t_bytes_key)

    t_int_key = dict[int, SimpleValue]
    value_int_key = {1: SimpleValue("val1"), 2: SimpleValue("val2")}
    validate_full_roundtrip(value_int_key, t_int_key)

    t_bool_key = dict[bool, SimpleValue]
    value_bool_key = {True: SimpleValue("val_true"), False: SimpleValue("val_false")}
    validate_full_roundtrip(value_bool_key, t_bool_key)

    t_str_key = dict[str, Order]
    value_str_key = {"K1": Order("O1", "item1", 10.0), "K2": Order("O2", "item2", 20.0)}
    validate_full_roundtrip(value_str_key, t_str_key)

    t_nt = dict[str, OrderNamedTuple]
    value_nt = {
        "K1": OrderNamedTuple("O1", "item1", 10.0),
        "K2": OrderNamedTuple("O2", "item2", 20.0),
    }
    validate_full_roundtrip(value_nt, t_nt)

    t_range_key = dict[cocoindex.Range, SimpleValue]
    value_range_key = {
        (1, 10): SimpleValue("val_range1"),
        (20, 30): SimpleValue("val_range2"),
    }
    validate_full_roundtrip(value_range_key, t_range_key)

    t_date_key = dict[datetime.date, SimpleValue]
    value_date_key = {
        datetime.date(2023, 1, 1): SimpleValue("val_date1"),
        datetime.date(2024, 2, 2): SimpleValue("val_date2"),
    }
    validate_full_roundtrip(value_date_key, t_date_key)

    t_uuid_key = dict[uuid.UUID, SimpleValue]
    value_uuid_key = {
        uuid.uuid4(): SimpleValue("val_uuid1"),
        uuid.uuid4(): SimpleValue("val_uuid2"),
    }
    validate_full_roundtrip(value_uuid_key, t_uuid_key)


def test_roundtrip_ktable_struct_key() -> None:
    @dataclass(frozen=True)
    class OrderKey:
        shop_id: str
        version: int

    t = dict[OrderKey, Order]
    value = {
        OrderKey("A", 3): Order("O1", "item1", 10.0),
        OrderKey("B", 4): Order("O2", "item2", 20.0),
    }
    validate_full_roundtrip(value, t)

    t_nt = dict[OrderKey, OrderNamedTuple]
    value_nt = {
        OrderKey("A", 3): OrderNamedTuple("O1", "item1", 10.0),
        OrderKey("B", 4): OrderNamedTuple("O2", "item2", 20.0),
    }
    validate_full_roundtrip(value_nt, t_nt)


IntVectorType = cocoindex.Vector[np.int64, Literal[5]]


def test_vector_as_vector() -> None:
    value = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    encoded = encode_engine_value(value, IntVectorType)
    assert np.array_equal(encoded, value)
    decoded = build_engine_value_decoder(IntVectorType)(encoded)
    assert np.array_equal(decoded, value)


ListIntType = list[int]


def test_vector_as_list() -> None:
    value: ListIntType = [1, 2, 3, 4, 5]
    encoded = encode_engine_value(value, ListIntType)
    assert encoded == [1, 2, 3, 4, 5]
    decoded = build_engine_value_decoder(ListIntType)(encoded)
    assert np.array_equal(decoded, value)


Float64VectorTypeNoDim = Vector[np.float64]
Float32VectorType = Vector[np.float32, Literal[3]]
Float64VectorType = Vector[np.float64, Literal[3]]
Int64VectorType = Vector[np.int64, Literal[3]]
NDArrayFloat32Type = NDArray[np.float32]
NDArrayFloat64Type = NDArray[np.float64]
NDArrayInt64Type = NDArray[np.int64]


def test_encode_engine_value_ndarray() -> None:
    """Test encoding NDArray vectors to lists for the Rust engine."""
    vec_f32: Float32VectorType = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(
        encode_engine_value(vec_f32, Float32VectorType), [1.0, 2.0, 3.0]
    )
    vec_f64: Float64VectorType = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert np.array_equal(
        encode_engine_value(vec_f64, Float64VectorType), [1.0, 2.0, 3.0]
    )
    vec_i64: Int64VectorType = np.array([1, 2, 3], dtype=np.int64)
    assert np.array_equal(encode_engine_value(vec_i64, Int64VectorType), [1, 2, 3])
    vec_nd_f32: NDArrayFloat32Type = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.array_equal(
        encode_engine_value(vec_nd_f32, NDArrayFloat32Type), [1.0, 2.0, 3.0]
    )


def test_make_engine_value_decoder_ndarray() -> None:
    """Test decoding engine lists to NDArray vectors."""
    decoder_f32 = build_engine_value_decoder(Float32VectorType)
    result_f32 = decoder_f32([1.0, 2.0, 3.0])
    assert isinstance(result_f32, np.ndarray)
    assert result_f32.dtype == np.float32
    assert np.array_equal(result_f32, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    decoder_f64 = build_engine_value_decoder(Float64VectorType)
    result_f64 = decoder_f64([1.0, 2.0, 3.0])
    assert isinstance(result_f64, np.ndarray)
    assert result_f64.dtype == np.float64
    assert np.array_equal(result_f64, np.array([1.0, 2.0, 3.0], dtype=np.float64))
    decoder_i64 = build_engine_value_decoder(Int64VectorType)
    result_i64 = decoder_i64([1, 2, 3])
    assert isinstance(result_i64, np.ndarray)
    assert result_i64.dtype == np.int64
    assert np.array_equal(result_i64, np.array([1, 2, 3], dtype=np.int64))
    decoder_nd_f32 = build_engine_value_decoder(NDArrayFloat32Type)
    result_nd_f32 = decoder_nd_f32([1.0, 2.0, 3.0])
    assert isinstance(result_nd_f32, np.ndarray)
    assert result_nd_f32.dtype == np.float32
    assert np.array_equal(result_nd_f32, np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_roundtrip_ndarray_vector() -> None:
    """Test roundtrip encoding and decoding of NDArray vectors."""
    value_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    encoded_f32 = encode_engine_value(value_f32, Float32VectorType)
    np.array_equal(encoded_f32, [1.0, 2.0, 3.0])
    decoded_f32 = build_engine_value_decoder(Float32VectorType)(encoded_f32)
    assert isinstance(decoded_f32, np.ndarray)
    assert decoded_f32.dtype == np.float32
    assert np.array_equal(decoded_f32, value_f32)
    value_i64 = np.array([1, 2, 3], dtype=np.int64)
    encoded_i64 = encode_engine_value(value_i64, Int64VectorType)
    assert np.array_equal(encoded_i64, [1, 2, 3])
    decoded_i64 = build_engine_value_decoder(Int64VectorType)(encoded_i64)
    assert isinstance(decoded_i64, np.ndarray)
    assert decoded_i64.dtype == np.int64
    assert np.array_equal(decoded_i64, value_i64)
    value_nd_f64: NDArrayFloat64Type = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    encoded_nd_f64 = encode_engine_value(value_nd_f64, NDArrayFloat64Type)
    assert np.array_equal(encoded_nd_f64, [1.0, 2.0, 3.0])
    decoded_nd_f64 = build_engine_value_decoder(NDArrayFloat64Type)(encoded_nd_f64)
    assert isinstance(decoded_nd_f64, np.ndarray)
    assert decoded_nd_f64.dtype == np.float64
    assert np.array_equal(decoded_nd_f64, value_nd_f64)


def test_ndarray_dimension_mismatch() -> None:
    """Test dimension enforcement for Vector with specified dimension."""
    value = np.array([1.0, 2.0], dtype=np.float32)
    encoded = encode_engine_value(value, NDArray[np.float32])
    assert np.array_equal(encoded, [1.0, 2.0])
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        build_engine_value_decoder(Float32VectorType)(encoded)


def test_list_vector_backward_compatibility() -> None:
    """Test that list-based vectors still work for backward compatibility."""
    value = [1, 2, 3, 4, 5]
    encoded = encode_engine_value(value, list[int])
    assert encoded == [1, 2, 3, 4, 5]
    decoded = build_engine_value_decoder(IntVectorType)(encoded)
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == np.int64
    assert np.array_equal(decoded, np.array([1, 2, 3, 4, 5], dtype=np.int64))
    value_list: ListIntType = [1, 2, 3, 4, 5]
    encoded = encode_engine_value(value_list, ListIntType)
    assert np.array_equal(encoded, [1, 2, 3, 4, 5])
    decoded = build_engine_value_decoder(ListIntType)(encoded)
    assert np.array_equal(decoded, [1, 2, 3, 4, 5])


def test_encode_complex_structure_with_ndarray() -> None:
    """Test encoding a complex structure that includes an NDArray."""

    @dataclass
    class MyStructWithNDArray:
        name: str
        data: NDArray[np.float32]
        value: int

    original = MyStructWithNDArray(
        name="test_np", data=np.array([1.0, 0.5], dtype=np.float32), value=100
    )
    encoded = encode_engine_value(original, MyStructWithNDArray)

    assert encoded[0] == original.name
    assert np.array_equal(encoded[1], original.data)
    assert encoded[2] == original.value


def test_decode_nullable_ndarray_none_or_value_input() -> None:
    """Test decoding a nullable NDArray with None or value inputs."""
    src_type_dict = decode_engine_value_type(
        {
            "kind": "Vector",
            "element_type": {"kind": "Float32"},
            "dimension": None,
        }
    )
    dst_annotation = NDArrayFloat32Type | None
    decoder = make_engine_value_decoder(
        [], src_type_dict, analyze_type_info(dst_annotation)
    )

    none_engine_value = None
    decoded_array = decoder(none_engine_value)
    assert decoded_array is None

    engine_value = [1.0, 2.0, 3.0]
    decoded_array = decoder(engine_value)

    assert isinstance(decoded_array, np.ndarray)
    assert decoded_array.dtype == np.float32
    np.testing.assert_array_equal(
        decoded_array, np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )


def test_decode_vector_string() -> None:
    """Test decoding a vector of strings works for Python native list type."""
    src_type_dict = decode_engine_value_type(
        {
            "kind": "Vector",
            "element_type": {"kind": "Str"},
            "dimension": None,
        }
    )
    decoder = make_engine_value_decoder(
        [], src_type_dict, analyze_type_info(Vector[str])
    )
    assert decoder(["hello", "world"]) == ["hello", "world"]


def test_decode_error_non_nullable_or_non_list_vector() -> None:
    """Test decoding errors for non-nullable vectors or non-list inputs."""
    src_type_dict = decode_engine_value_type(
        {
            "kind": "Vector",
            "element_type": {"kind": "Float32"},
            "dimension": None,
        }
    )
    decoder = make_engine_value_decoder(
        [], src_type_dict, analyze_type_info(NDArrayFloat32Type)
    )
    with pytest.raises(ValueError, match="Received null for non-nullable vector"):
        decoder(None)
    with pytest.raises(TypeError, match="Expected NDArray or list for vector"):
        decoder("not a list")


def test_full_roundtrip_vector_numeric_types() -> None:
    """Test full roundtrip for numeric vector types using NDArray."""
    value_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    validate_full_roundtrip(
        value_f32,
        Vector[np.float32, Literal[3]],
        ([np.float32(1.0), np.float32(2.0), np.float32(3.0)], list[np.float32]),
        ([1.0, 2.0, 3.0], list[cocoindex.Float32]),
        ([1.0, 2.0, 3.0], list[float]),
    )
    validate_full_roundtrip(
        value_f32,
        np.typing.NDArray[np.float32],
        ([np.float32(1.0), np.float32(2.0), np.float32(3.0)], list[np.float32]),
        ([1.0, 2.0, 3.0], list[cocoindex.Float32]),
        ([1.0, 2.0, 3.0], list[float]),
    )
    validate_full_roundtrip(
        value_f32.tolist(),
        list[np.float32],
        (value_f32, Vector[np.float32, Literal[3]]),
        ([1.0, 2.0, 3.0], list[cocoindex.Float32]),
        ([1.0, 2.0, 3.0], list[float]),
    )

    value_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    validate_full_roundtrip(
        value_f64,
        Vector[np.float64, Literal[3]],
        ([np.float64(1.0), np.float64(2.0), np.float64(3.0)], list[np.float64]),
        ([1.0, 2.0, 3.0], list[cocoindex.Float64]),
        ([1.0, 2.0, 3.0], list[float]),
    )

    value_i64 = np.array([1, 2, 3], dtype=np.int64)
    validate_full_roundtrip(
        value_i64,
        Vector[np.int64, Literal[3]],
        ([np.int64(1), np.int64(2), np.int64(3)], list[np.int64]),
        ([1, 2, 3], list[int]),
    )

    value_i32 = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        validate_full_roundtrip(value_i32, Vector[np.int32, Literal[3]])
    value_u8 = np.array([1, 2, 3], dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        validate_full_roundtrip(value_u8, Vector[np.uint8, Literal[3]])
    value_u16 = np.array([1, 2, 3], dtype=np.uint16)
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        validate_full_roundtrip(value_u16, Vector[np.uint16, Literal[3]])
    value_u32 = np.array([1, 2, 3], dtype=np.uint32)
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        validate_full_roundtrip(value_u32, Vector[np.uint32, Literal[3]])
    value_u64 = np.array([1, 2, 3], dtype=np.uint64)
    with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
        validate_full_roundtrip(value_u64, Vector[np.uint64, Literal[3]])


def test_full_roundtrip_vector_of_vector() -> None:
    """Test full roundtrip for vector of vector."""
    value_f32 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    validate_full_roundtrip(
        value_f32,
        Vector[Vector[np.float32, Literal[3]], Literal[2]],
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], list[list[np.float32]]),
        ([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], list[list[cocoindex.Float32]]),
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            list[Vector[cocoindex.Float32, Literal[3]]],
        ),
        (
            value_f32,
            np.typing.NDArray[np.float32],
        ),
    )


def test_full_roundtrip_vector_other_types() -> None:
    """Test full roundtrip for Vector with non-numeric basic types."""
    uuid_list = [uuid.uuid4(), uuid.uuid4()]
    validate_full_roundtrip(uuid_list, Vector[uuid.UUID], (uuid_list, list[uuid.UUID]))

    date_list = [datetime.date(2023, 1, 1), datetime.date(2024, 10, 5)]
    validate_full_roundtrip(
        date_list, Vector[datetime.date], (date_list, list[datetime.date])
    )

    bool_list = [True, False, True, False]
    validate_full_roundtrip(bool_list, Vector[bool], (bool_list, list[bool]))

    validate_full_roundtrip([], Vector[uuid.UUID], ([], list[uuid.UUID]))
    validate_full_roundtrip([], Vector[datetime.date], ([], list[datetime.date]))
    validate_full_roundtrip([], Vector[bool], ([], list[bool]))


def test_roundtrip_vector_no_dimension() -> None:
    """Test full roundtrip for vector types without dimension annotation."""
    value_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    validate_full_roundtrip(
        value_f64,
        Vector[np.float64],
        ([1.0, 2.0, 3.0], list[float]),
        (np.array([1.0, 2.0, 3.0], dtype=np.float64), np.typing.NDArray[np.float64]),
    )


def test_roundtrip_string_vector() -> None:
    """Test full roundtrip for string vector using list."""
    value_str: Vector[str] = ["hello", "world"]
    validate_full_roundtrip(value_str, Vector[str])


def test_roundtrip_empty_vector() -> None:
    """Test full roundtrip for empty numeric vector."""
    value_empty: Vector[np.float32] = np.array([], dtype=np.float32)
    validate_full_roundtrip(value_empty, Vector[np.float32])


def test_roundtrip_dimension_mismatch() -> None:
    """Test that dimension mismatch raises an error during roundtrip."""
    value_f32: Vector[np.float32, Literal[3]] = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        validate_full_roundtrip(value_f32, Vector[np.float32, Literal[3]])


def test_full_roundtrip_scalar_numeric_types() -> None:
    """Test full roundtrip for scalar NumPy numeric types."""
    # Test supported scalar types
    validate_full_roundtrip(np.int64(42), np.int64, (42, int))
    validate_full_roundtrip(np.float32(3.25), np.float32, (3.25, cocoindex.Float32))
    validate_full_roundtrip(np.float64(3.25), np.float64, (3.25, cocoindex.Float64))

    # Test unsupported scalar types
    for unsupported_type in [np.int32, np.uint8, np.uint16, np.uint32, np.uint64]:
        with pytest.raises(ValueError, match="Unsupported NumPy dtype"):
            validate_full_roundtrip(unsupported_type(1), unsupported_type)


def test_full_roundtrip_nullable_scalar() -> None:
    """Test full roundtrip for nullable scalar NumPy types."""
    # Test with non-null values
    validate_full_roundtrip(np.int64(42), np.int64 | None)
    validate_full_roundtrip(np.float32(3.14), np.float32 | None)
    validate_full_roundtrip(np.float64(2.718), np.float64 | None)

    # Test with None
    validate_full_roundtrip(None, np.int64 | None)
    validate_full_roundtrip(None, np.float32 | None)
    validate_full_roundtrip(None, np.float64 | None)


def test_full_roundtrip_scalar_in_struct() -> None:
    """Test full roundtrip for scalar NumPy types in a dataclass."""

    @dataclass
    class NumericStruct:
        int_field: np.int64
        float32_field: np.float32
        float64_field: np.float64

    instance = NumericStruct(
        int_field=np.int64(42),
        float32_field=np.float32(3.14),
        float64_field=np.float64(2.718),
    )
    validate_full_roundtrip(instance, NumericStruct)


def test_full_roundtrip_scalar_in_nested_struct() -> None:
    """Test full roundtrip for scalar NumPy types in a nested struct."""

    @dataclass
    class InnerStruct:
        value: np.float64

    @dataclass
    class OuterStruct:
        inner: InnerStruct
        count: np.int64

    instance = OuterStruct(
        inner=InnerStruct(value=np.float64(2.718)),
        count=np.int64(1),
    )
    validate_full_roundtrip(instance, OuterStruct)


def test_full_roundtrip_scalar_with_python_types() -> None:
    """Test full roundtrip for structs mixing NumPy and Python scalar types."""

    @dataclass
    class MixedStruct:
        numpy_int: np.int64
        python_int: int
        numpy_float: np.float64
        python_float: float
        string: str
        annotated_int: Annotated[np.int64, TypeKind("Int64")]
        annotated_float: Float32

    instance = MixedStruct(
        numpy_int=np.int64(42),
        python_int=43,
        numpy_float=np.float64(2.718),
        python_float=3.14,
        string="hello, world",
        annotated_int=np.int64(42),
        annotated_float=2.0,
    )
    validate_full_roundtrip(instance, MixedStruct)


def test_roundtrip_simple_struct_to_dict_binding() -> None:
    """Test struct -> dict binding with Any annotation."""

    @dataclass
    class SimpleStruct:
        first_name: str
        last_name: str

    instance = SimpleStruct("John", "Doe")
    expected_dict = {"first_name": "John", "last_name": "Doe"}

    # Test Any annotation
    validate_full_roundtrip(
        instance,
        SimpleStruct,
        (expected_dict, Any),
        (expected_dict, dict),
        (expected_dict, dict[Any, Any]),
        (expected_dict, dict[str, Any]),
        # For simple struct, all fields have the same type, so we can directly use the type as the dict value type.
        (expected_dict, dict[Any, str]),
        (expected_dict, dict[str, str]),
    )

    with pytest.raises(ValueError):
        validate_full_roundtrip(instance, SimpleStruct, (expected_dict, dict[str, int]))

    with pytest.raises(ValueError):
        validate_full_roundtrip(instance, SimpleStruct, (expected_dict, dict[int, Any]))


def test_roundtrip_struct_to_dict_binding() -> None:
    """Test struct -> dict binding with Any annotation."""

    @dataclass
    class SimpleStruct:
        name: str
        value: int
        price: float

    instance = SimpleStruct("test", 42, 3.14)
    expected_dict = {"name": "test", "value": 42, "price": 3.14}

    # Test Any annotation
    validate_full_roundtrip(
        instance,
        SimpleStruct,
        (expected_dict, Any),
        (expected_dict, dict),
        (expected_dict, dict[Any, Any]),
        (expected_dict, dict[str, Any]),
    )

    with pytest.raises(ValueError):
        validate_full_roundtrip(instance, SimpleStruct, (expected_dict, dict[str, str]))

    with pytest.raises(ValueError):
        validate_full_roundtrip(instance, SimpleStruct, (expected_dict, dict[int, Any]))


def test_roundtrip_struct_to_dict_explicit() -> None:
    """Test struct -> dict binding with explicit dict annotations."""

    @dataclass
    class Product:
        id: str
        name: str
        price: float
        active: bool

    instance = Product("P1", "Widget", 29.99, True)
    expected_dict = {"id": "P1", "name": "Widget", "price": 29.99, "active": True}

    # Test explicit dict annotations
    validate_full_roundtrip(
        instance, Product, (expected_dict, dict), (expected_dict, dict[str, Any])
    )


def test_roundtrip_struct_to_dict_with_none_annotation() -> None:
    """Test struct -> dict binding with None annotation."""

    @dataclass
    class Config:
        host: str
        port: int
        debug: bool

    instance = Config("localhost", 8080, True)
    expected_dict = {"host": "localhost", "port": 8080, "debug": True}

    # Test empty annotation (should be treated as Any)
    validate_full_roundtrip(instance, Config, (expected_dict, inspect.Parameter.empty))


def test_roundtrip_struct_to_dict_nested() -> None:
    """Test struct -> dict binding with nested structs."""

    @dataclass
    class Address:
        street: str
        city: str

    @dataclass
    class Person:
        name: str
        age: int
        address: Address

    address = Address("123 Main St", "Anytown")
    person = Person("John", 30, address)
    expected_dict = {
        "name": "John",
        "age": 30,
        "address": {"street": "123 Main St", "city": "Anytown"},
    }

    # Test nested struct conversion
    validate_full_roundtrip(person, Person, (expected_dict, dict[str, Any]))


def test_roundtrip_struct_to_dict_with_list() -> None:
    """Test struct -> dict binding with list fields."""

    @dataclass
    class Team:
        name: str
        members: list[str]
        active: bool

    instance = Team("Dev Team", ["Alice", "Bob", "Charlie"], True)
    expected_dict = {
        "name": "Dev Team",
        "members": ["Alice", "Bob", "Charlie"],
        "active": True,
    }

    validate_full_roundtrip(instance, Team, (expected_dict, dict))


def test_roundtrip_namedtuple_to_dict_binding() -> None:
    """Test NamedTuple -> dict binding."""

    class Point(NamedTuple):
        x: float
        y: float
        z: float

    instance = Point(1.0, 2.0, 3.0)
    expected_dict = {"x": 1.0, "y": 2.0, "z": 3.0}

    validate_full_roundtrip(
        instance, Point, (expected_dict, dict), (expected_dict, Any)
    )


def test_roundtrip_ltable_to_list_dict_binding() -> None:
    """Test LTable -> list[dict] binding with Any annotation."""

    @dataclass
    class User:
        id: str
        name: str
        age: int

    users = [User("u1", "Alice", 25), User("u2", "Bob", 30), User("u3", "Charlie", 35)]
    expected_list_dict = [
        {"id": "u1", "name": "Alice", "age": 25},
        {"id": "u2", "name": "Bob", "age": 30},
        {"id": "u3", "name": "Charlie", "age": 35},
    ]

    # Test Any annotation
    validate_full_roundtrip(
        users,
        list[User],
        (expected_list_dict, Any),
        (expected_list_dict, list[Any]),
        (expected_list_dict, list[dict[str, Any]]),
    )


def test_roundtrip_ktable_to_dict_dict_binding() -> None:
    """Test KTable -> dict[K, dict] binding with Any annotation."""

    @dataclass
    class Product:
        name: str
        price: float
        active: bool

    products = {
        "p1": Product("Widget", 29.99, True),
        "p2": Product("Gadget", 49.99, False),
        "p3": Product("Tool", 19.99, True),
    }
    expected_dict_dict = {
        "p1": {"name": "Widget", "price": 29.99, "active": True},
        "p2": {"name": "Gadget", "price": 49.99, "active": False},
        "p3": {"name": "Tool", "price": 19.99, "active": True},
    }

    # Test Any annotation
    validate_full_roundtrip(
        products,
        dict[str, Product],
        (expected_dict_dict, Any),
        (expected_dict_dict, dict),
        (expected_dict_dict, dict[Any, Any]),
        (expected_dict_dict, dict[str, Any]),
        (expected_dict_dict, dict[Any, dict[Any, Any]]),
        (expected_dict_dict, dict[str, dict[Any, Any]]),
        (expected_dict_dict, dict[str, dict[str, Any]]),
    )


def test_roundtrip_ktable_with_complex_key() -> None:
    """Test KTable with complex key types -> dict binding."""

    @dataclass(frozen=True)
    class OrderKey:
        shop_id: str
        version: int

    @dataclass
    class Order:
        customer: str
        total: float

    orders = {
        OrderKey("shop1", 1): Order("Alice", 100.0),
        OrderKey("shop2", 2): Order("Bob", 200.0),
    }
    expected_dict_dict = {
        ("shop1", 1): {"customer": "Alice", "total": 100.0},
        ("shop2", 2): {"customer": "Bob", "total": 200.0},
    }

    # Test Any annotation
    validate_full_roundtrip(
        orders,
        dict[OrderKey, Order],
        (expected_dict_dict, Any),
        (expected_dict_dict, dict),
        (expected_dict_dict, dict[Any, Any]),
        (expected_dict_dict, dict[Any, dict[str, Any]]),
        (
            {
                ("shop1", 1): Order("Alice", 100.0),
                ("shop2", 2): Order("Bob", 200.0),
            },
            dict[Any, Order],
        ),
        (
            {
                OrderKey("shop1", 1): {"customer": "Alice", "total": 100.0},
                OrderKey("shop2", 2): {"customer": "Bob", "total": 200.0},
            },
            dict[OrderKey, Any],
        ),
    )


def test_roundtrip_ltable_with_nested_structs() -> None:
    """Test LTable with nested structs -> list[dict] binding."""

    @dataclass
    class Address:
        street: str
        city: str

    @dataclass
    class Person:
        name: str
        age: int
        address: Address

    people = [
        Person("John", 30, Address("123 Main St", "Anytown")),
        Person("Jane", 25, Address("456 Oak Ave", "Somewhere")),
    ]
    expected_list_dict = [
        {
            "name": "John",
            "age": 30,
            "address": {"street": "123 Main St", "city": "Anytown"},
        },
        {
            "name": "Jane",
            "age": 25,
            "address": {"street": "456 Oak Ave", "city": "Somewhere"},
        },
    ]

    # Test Any annotation
    validate_full_roundtrip(people, list[Person], (expected_list_dict, Any))


def test_roundtrip_ktable_with_list_fields() -> None:
    """Test KTable with list fields -> dict binding."""

    @dataclass
    class Team:
        name: str
        members: list[str]
        active: bool

    teams = {
        "team1": Team("Dev Team", ["Alice", "Bob"], True),
        "team2": Team("QA Team", ["Charlie", "David"], False),
    }
    expected_dict_dict = {
        "team1": {"name": "Dev Team", "members": ["Alice", "Bob"], "active": True},
        "team2": {"name": "QA Team", "members": ["Charlie", "David"], "active": False},
    }

    # Test Any annotation
    validate_full_roundtrip(teams, dict[str, Team], (expected_dict_dict, Any))


def test_auto_default_for_supported_and_unsupported_types() -> None:
    @dataclass
    class Base:
        a: int

    @dataclass
    class NullableField:
        a: int
        b: int | None

    @dataclass
    class LTableField:
        a: int
        b: list[Base]

    @dataclass
    class KTableField:
        a: int
        b: dict[str, Base]

    @dataclass
    class UnsupportedField:
        a: int
        b: int

    validate_full_roundtrip(NullableField(1, None), NullableField)

    validate_full_roundtrip(LTableField(1, []), LTableField)

    validate_full_roundtrip(KTableField(1, {}), KTableField)

    with pytest.raises(
        ValueError,
        match=r"Field 'b' \(type <class 'int'>\) without default value is missing in input: ",
    ):
        build_engine_value_decoder(Base, UnsupportedField)


# Pydantic model tests
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_simple_struct() -> None:
    """Test basic Pydantic model encoding and decoding."""
    order = OrderPydantic(order_id="O1", name="item1", price=10.0)
    validate_full_roundtrip(order, OrderPydantic)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_struct_with_defaults() -> None:
    """Test Pydantic model with default values."""
    order = OrderPydantic(order_id="O1", name="item1", price=10.0)
    assert order.extra_field == "default_extra"
    validate_full_roundtrip(order, OrderPydantic)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_nested_struct() -> None:
    """Test nested Pydantic models."""
    order = OrderPydantic(order_id="O1", name="item1", price=10.0)
    customer = CustomerPydantic(name="Alice", order=order)
    validate_full_roundtrip(customer, CustomerPydantic)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_struct_with_list() -> None:
    """Test Pydantic model with list fields."""
    order = OrderPydantic(order_id="O1", name="item1", price=10.0)
    tags = [TagPydantic(name="vip"), TagPydantic(name="premium")]
    customer = CustomerPydantic(name="Alice", order=order, tags=tags)
    validate_full_roundtrip(customer, CustomerPydantic)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_complex_nested_struct() -> None:
    """Test complex nested Pydantic structure."""
    order1 = OrderPydantic(order_id="O1", name="item1", price=10.0)
    order2 = OrderPydantic(order_id="O2", name="item2", price=20.0)
    customer = CustomerPydantic(
        name="Alice", order=order1, tags=[TagPydantic(name="vip")]
    )
    nested = NestedStructPydantic(customer=customer, orders=[order1, order2], count=2)
    validate_full_roundtrip(nested, NestedStructPydantic)


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_struct_to_dict_binding() -> None:
    """Test Pydantic model -> dict binding."""
    order = OrderPydantic(order_id="O1", name="item1", price=10.0, extra_field="custom")
    expected_dict = {
        "order_id": "O1",
        "name": "item1",
        "price": 10.0,
        "extra_field": "custom",
    }

    validate_full_roundtrip(
        order,
        OrderPydantic,
        (expected_dict, Any),
        (expected_dict, dict),
        (expected_dict, dict[Any, Any]),
        (expected_dict, dict[str, Any]),
    )


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_make_engine_value_decoder_pydantic_struct() -> None:
    """Test engine value decoder for Pydantic models."""
    engine_val = ["O1", "item1", 10.0, "default_extra"]
    decoder = build_engine_value_decoder(OrderPydantic)
    result = decoder(engine_val)

    assert isinstance(result, OrderPydantic)
    assert result.order_id == "O1"
    assert result.name == "item1"
    assert result.price == 10.0
    assert result.extra_field == "default_extra"


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_make_engine_value_decoder_pydantic_nested() -> None:
    """Test engine value decoder for nested Pydantic models."""
    engine_val = [
        "Alice",
        ["O1", "item1", 10.0, "default_extra"],
        [["vip"]],
    ]
    decoder = build_engine_value_decoder(CustomerPydantic)
    result = decoder(engine_val)

    assert isinstance(result, CustomerPydantic)
    assert result.name == "Alice"
    assert isinstance(result.order, OrderPydantic)
    assert result.order.order_id == "O1"
    assert result.tags is not None
    assert len(result.tags) == 1
    assert isinstance(result.tags[0], TagPydantic)
    assert result.tags[0].name == "vip"


@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_pydantic_mixed_with_dataclass() -> None:
    """Test mixing Pydantic models with dataclasses."""

    # Create a dataclass that uses a Pydantic model
    @dataclass
    class MixedStruct:
        name: str
        pydantic_order: OrderPydantic

    order = OrderPydantic(order_id="O1", name="item1", price=10.0)
    mixed = MixedStruct(name="test", pydantic_order=order)
    validate_full_roundtrip(mixed, MixedStruct)
