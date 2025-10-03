use crate::prelude::*;

use crate::utils::immutable::RefList;
use schemars::schema::{
    ArrayValidation, InstanceType, ObjectValidation, Schema, SchemaObject, SingleOrVec,
    SubschemaValidation,
};
use std::fmt::Write;

pub struct ToJsonSchemaOptions {
    /// If true, mark all fields as required.
    /// Use union type (with `null`) for optional fields instead.
    /// Models like OpenAI will reject the schema if a field is not required.
    pub fields_always_required: bool,

    /// If true, the JSON schema supports the `format` keyword.
    pub supports_format: bool,

    /// If true, extract descriptions to a separate extra instruction.
    pub extract_descriptions: bool,

    /// If true, the top level must be a JSON object.
    pub top_level_must_be_object: bool,
}

struct JsonSchemaBuilder {
    options: ToJsonSchemaOptions,
    extra_instructions_per_field: IndexMap<String, String>,
}

impl JsonSchemaBuilder {
    fn new(options: ToJsonSchemaOptions) -> Self {
        Self {
            options,
            extra_instructions_per_field: IndexMap::new(),
        }
    }

    fn add_description(
        &mut self,
        schema: &mut SchemaObject,
        description: &str,
        field_path: RefList<'_, &'_ spec::FieldName>,
    ) {
        let mut_description = if self.options.extract_descriptions {
            let mut fields: Vec<_> = field_path.iter().map(|f| f.as_str()).collect();
            fields.reverse();
            let field_path_str = fields.join(".");

            self.extra_instructions_per_field
                .entry(field_path_str)
                .or_default()
        } else {
            schema
                .metadata
                .get_or_insert_default()
                .description
                .get_or_insert_default()
        };
        if !mut_description.is_empty() {
            mut_description.push_str("\n\n");
        }
        mut_description.push_str(description);
    }

    fn for_basic_value_type(
        &mut self,
        schema_base: SchemaObject,
        basic_type: &schema::BasicValueType,
        field_path: RefList<'_, &'_ spec::FieldName>,
    ) -> SchemaObject {
        let mut schema = schema_base;
        match basic_type {
            schema::BasicValueType::Str => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
            }
            schema::BasicValueType::Bytes => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
            }
            schema::BasicValueType::Bool => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Boolean)));
            }
            schema::BasicValueType::Int64 => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Integer)));
            }
            schema::BasicValueType::Float32 | schema::BasicValueType::Float64 => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Number)));
            }
            schema::BasicValueType::Range => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Array)));
                schema.array = Some(Box::new(ArrayValidation {
                    items: Some(SingleOrVec::Single(Box::new(
                        SchemaObject {
                            instance_type: Some(SingleOrVec::Single(Box::new(
                                InstanceType::Integer,
                            ))),
                            ..Default::default()
                        }
                        .into(),
                    ))),
                    min_items: Some(2),
                    max_items: Some(2),
                    ..Default::default()
                }));
                self.add_description(
                    &mut schema,
                    "A range represented by a list of two positions, start pos (inclusive), end pos (exclusive).",
                    field_path,
                );
            }
            schema::BasicValueType::Uuid => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("uuid".to_string());
                }
                self.add_description(
                    &mut schema,
                    "A UUID, e.g. 123e4567-e89b-12d3-a456-426614174000",
                    field_path,
                );
            }
            schema::BasicValueType::Date => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("date".to_string());
                }
                self.add_description(
                    &mut schema,
                    "A date in YYYY-MM-DD format, e.g. 2025-03-27",
                    field_path,
                );
            }
            schema::BasicValueType::Time => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("time".to_string());
                }
                self.add_description(
                    &mut schema,
                    "A time in HH:MM:SS format, e.g. 13:32:12",
                    field_path,
                );
            }
            schema::BasicValueType::LocalDateTime => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("date-time".to_string());
                }
                self.add_description(
                    &mut schema,
                    "Date time without timezone offset in YYYY-MM-DDTHH:MM:SS format, e.g. 2025-03-27T13:32:12",
                    field_path,
                );
            }
            schema::BasicValueType::OffsetDateTime => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("date-time".to_string());
                }
                self.add_description(
                    &mut schema,
                    "Date time with timezone offset in RFC3339, e.g. 2025-03-27T13:32:12Z, 2025-03-27T07:32:12.313-06:00",
                    field_path,
                );
            }
            &schema::BasicValueType::TimeDelta => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::String)));
                if self.options.supports_format {
                    schema.format = Some("duration".to_string());
                }
                self.add_description(
                    &mut schema,
                    "A duration, e.g. 'PT1H2M3S' (ISO 8601) or '1 day 2 hours 3 seconds'",
                    field_path,
                );
            }
            schema::BasicValueType::Json => {
                // Can be any value. No type constraint.
            }
            schema::BasicValueType::Vector(s) => {
                schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Array)));
                schema.array = Some(Box::new(ArrayValidation {
                    items: Some(SingleOrVec::Single(Box::new(
                        self.for_basic_value_type(
                            SchemaObject::default(),
                            &s.element_type,
                            field_path,
                        )
                        .into(),
                    ))),
                    min_items: s.dimension.and_then(|d| u32::try_from(d).ok()),
                    max_items: s.dimension.and_then(|d| u32::try_from(d).ok()),
                    ..Default::default()
                }));
            }
            schema::BasicValueType::Union(s) => {
                schema.subschemas = Some(Box::new(SubschemaValidation {
                    one_of: Some(
                        s.types
                            .iter()
                            .map(|t| {
                                Schema::Object(self.for_basic_value_type(
                                    SchemaObject::default(),
                                    t,
                                    field_path,
                                ))
                            })
                            .collect(),
                    ),
                    ..Default::default()
                }));
            }
        }
        schema
    }

    fn for_struct_schema(
        &mut self,
        schema_base: SchemaObject,
        struct_schema: &schema::StructSchema,
        field_path: RefList<'_, &'_ spec::FieldName>,
    ) -> SchemaObject {
        let mut schema = schema_base;
        if let Some(description) = &struct_schema.description {
            self.add_description(&mut schema, description, field_path);
        }
        schema.instance_type = Some(SingleOrVec::Single(Box::new(InstanceType::Object)));
        schema.object = Some(Box::new(ObjectValidation {
            properties: struct_schema
                .fields
                .iter()
                .map(|f| {
                    let mut field_schema_base = SchemaObject::default();
                    // Set field description if available
                    if let Some(description) = &f.description {
                        self.add_description(
                            &mut field_schema_base,
                            description,
                            field_path.prepend(&f.name),
                        );
                    }
                    let mut field_schema = self.for_enriched_value_type(
                        field_schema_base,
                        &f.value_type,
                        field_path.prepend(&f.name),
                    );
                    if self.options.fields_always_required && f.value_type.nullable {
                        if let Some(instance_type) = &mut field_schema.instance_type {
                            let mut types = match instance_type {
                                SingleOrVec::Single(t) => vec![**t],
                                SingleOrVec::Vec(t) => std::mem::take(t),
                            };
                            types.push(InstanceType::Null);
                            *instance_type = SingleOrVec::Vec(types);
                        }
                    }
                    (f.name.to_string(), field_schema.into())
                })
                .collect(),
            required: struct_schema
                .fields
                .iter()
                .filter(|&f| self.options.fields_always_required || !f.value_type.nullable)
                .map(|f| f.name.to_string())
                .collect(),
            additional_properties: Some(Schema::Bool(false).into()),
            ..Default::default()
        }));
        schema
    }

    fn for_value_type(
        &mut self,
        schema_base: SchemaObject,
        value_type: &schema::ValueType,
        field_path: RefList<'_, &'_ spec::FieldName>,
    ) -> SchemaObject {
        match value_type {
            schema::ValueType::Basic(b) => self.for_basic_value_type(schema_base, b, field_path),
            schema::ValueType::Struct(s) => self.for_struct_schema(schema_base, s, field_path),
            schema::ValueType::Table(c) => SchemaObject {
                instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Array))),
                array: Some(Box::new(ArrayValidation {
                    items: Some(SingleOrVec::Single(Box::new(
                        self.for_struct_schema(SchemaObject::default(), &c.row, field_path)
                            .into(),
                    ))),
                    ..Default::default()
                })),
                ..schema_base
            },
        }
    }

    fn for_enriched_value_type(
        &mut self,
        schema_base: SchemaObject,
        enriched_value_type: &schema::EnrichedValueType,
        field_path: RefList<'_, &'_ spec::FieldName>,
    ) -> SchemaObject {
        self.for_value_type(schema_base, &enriched_value_type.typ, field_path)
    }

    fn build_extra_instructions(&self) -> Result<Option<String>> {
        if self.extra_instructions_per_field.is_empty() {
            return Ok(None);
        }

        let mut instructions = String::new();
        write!(&mut instructions, "Instructions for specific fields:\n\n")?;
        for (field_path, instruction) in self.extra_instructions_per_field.iter() {
            write!(
                &mut instructions,
                "- {}: {}\n\n",
                if field_path.is_empty() {
                    "(root object)"
                } else {
                    field_path.as_str()
                },
                instruction
            )?;
        }
        Ok(Some(instructions))
    }
}

pub struct ValueExtractor {
    value_type: schema::ValueType,
    object_wrapper_field_name: Option<String>,
}

impl ValueExtractor {
    pub fn extract_value(&self, json_value: serde_json::Value) -> Result<value::Value> {
        let unwrapped_json_value =
            if let Some(object_wrapper_field_name) = &self.object_wrapper_field_name {
                match json_value {
                    serde_json::Value::Object(mut o) => o
                        .remove(object_wrapper_field_name)
                        .unwrap_or(serde_json::Value::Null),
                    _ => {
                        bail!("Field `{}` not found", object_wrapper_field_name)
                    }
                }
            } else {
                json_value
            };
        let result = value::Value::from_json(unwrapped_json_value, &self.value_type)?;
        Ok(result)
    }
}

pub struct BuildJsonSchemaOutput {
    pub schema: SchemaObject,
    pub extra_instructions: Option<String>,
    pub value_extractor: ValueExtractor,
}

pub fn build_json_schema(
    value_type: schema::EnrichedValueType,
    options: ToJsonSchemaOptions,
) -> Result<BuildJsonSchemaOutput> {
    let mut builder = JsonSchemaBuilder::new(options);
    let (schema, object_wrapper_field_name) = if builder.options.top_level_must_be_object
        && !matches!(value_type.typ, schema::ValueType::Struct(_))
    {
        let object_wrapper_field_name = "value".to_string();
        let wrapper_struct = schema::StructSchema {
            fields: Arc::new(vec![schema::FieldSchema {
                name: object_wrapper_field_name.clone(),
                value_type: value_type.clone(),
                description: None,
            }]),
            description: None,
        };
        (
            builder.for_struct_schema(SchemaObject::default(), &wrapper_struct, RefList::Nil),
            Some(object_wrapper_field_name),
        )
    } else {
        (
            builder.for_enriched_value_type(SchemaObject::default(), &value_type, RefList::Nil),
            None,
        )
    };
    Ok(BuildJsonSchemaOutput {
        schema,
        extra_instructions: builder.build_extra_instructions()?,
        value_extractor: ValueExtractor {
            value_type: value_type.typ,
            object_wrapper_field_name,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::schema::{
        BasicValueType, EnrichedValueType, FieldSchema, StructSchema, ValueType,
    };
    use std::sync::Arc;

    #[test]
    fn test_description_concatenation() {
        // Create a struct with a field that has both field-level and type-level descriptions
        let struct_schema = StructSchema {
            description: Some(Arc::from("Test struct description")),
            fields: Arc::new(vec![FieldSchema {
                name: "uuid_field".to_string(),
                value_type: EnrichedValueType {
                    typ: ValueType::Basic(BasicValueType::Uuid),
                    nullable: false,
                    attrs: Default::default(),
                },
                description: Some(Arc::from("This is a field-level description for UUID")),
            }]),
        };

        let enriched_value_type = EnrichedValueType {
            typ: ValueType::Struct(struct_schema),
            nullable: false,
            attrs: Default::default(),
        };

        let options = ToJsonSchemaOptions {
            fields_always_required: false,
            supports_format: true,
            extract_descriptions: false, // We want to see the description in the schema
            top_level_must_be_object: false,
        };

        let result = build_json_schema(enriched_value_type, options).unwrap();

        // Check if the description contains both field and type descriptions
        if let Some(properties) = &result.schema.object
            && let Some(uuid_field_schema) = properties.properties.get("uuid_field")
            && let Schema::Object(schema_object) = uuid_field_schema
            && let Some(description) = &schema_object
                .metadata
                .as_ref()
                .and_then(|m| m.description.as_ref())
        {
            assert_eq!(
                description.as_str(),
                "This is a field-level description for UUID\n\nA UUID, e.g. 123e4567-e89b-12d3-a456-426614174000"
            );
        } else {
            panic!("No description found in the schema");
        }
    }
}
