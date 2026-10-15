"""Tests for the JSON Schema subset: validation and coercion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.schemas import (  # noqa: E402
    coerce,
    example_for,
    render_schema_hint,
    validate,
)

WEATHER = {
    "type": "object",
    "required": ["city"],
    "properties": {
        "city": {"type": "string"},
        "units": {"type": "string", "enum": ["metric", "imperial"]},
        "days": {"type": "integer"},
        "detailed": {"type": "boolean"},
    },
}


class TestValidate(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(validate({"city": "Paris"}, WEATHER), [])

    def test_missing_required(self):
        errors = validate({"units": "metric"}, WEATHER)
        self.assertTrue(any("city" in e for e in errors))

    def test_wrong_type(self):
        errors = validate({"city": "Paris", "days": "three"}, WEATHER)
        self.assertTrue(any("days" in e for e in errors))

    def test_enum_violation(self):
        errors = validate({"city": "Paris", "units": "kelvin"}, WEATHER)
        self.assertTrue(any("units" in e for e in errors))

    def test_additional_properties_false(self):
        schema = dict(WEATHER)
        schema["additionalProperties"] = False
        errors = validate({"city": "Paris", "bogus": 1}, schema)
        self.assertTrue(any("bogus" in e for e in errors))

    def test_bool_is_not_an_integer(self):
        errors = validate({"city": "Paris", "days": True}, WEATHER)
        self.assertTrue(any("days" in e for e in errors))

    def test_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "traveller": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                }
            },
        }
        errors = validate({"traveller": {"age": 3}}, schema)
        self.assertTrue(
            any("traveller" in e and "name" in e for e in errors), errors
        )

    def test_array_items(self):
        schema = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
        errors = validate({"tags": ["a", 2]}, schema)
        self.assertTrue(any("tags[1]" in e for e in errors))

    def test_minimum_and_maximum(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1, "maximum": 5}}}
        self.assertTrue(any("minimum" in e for e in validate({"n": 0}, schema)))
        self.assertTrue(any("maximum" in e for e in validate({"n": 9}, schema)))

    def test_pattern(self):
        schema = {"type": "object", "properties": {"code": {"type": "string", "pattern": "^[A-Z]{2}$"}}}
        self.assertTrue(validate({"code": "abc"}, schema))
        self.assertEqual(validate({"code": "AB"}, schema), [])

    def test_nullable(self):
        schema = {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
        self.assertEqual(validate({"x": None}, schema), [])

    def test_empty_schema_allows_anything(self):
        self.assertEqual(validate({"anything": 1}, {}), [])


class TestCoerce(unittest.TestCase):
    def test_string_to_integer(self):
        value, notes = coerce({"days": "3"}, WEATHER)
        self.assertEqual(value["days"], 3)
        self.assertTrue(any("integer" in n for n in notes))

    def test_string_to_number(self):
        schema = {"type": "object", "properties": {"x": {"type": "number"}}}
        value, notes = coerce({"x": "3.5"}, schema)
        self.assertEqual(value["x"], 3.5)
        self.assertTrue(notes)

    def test_string_to_boolean(self):
        value, notes = coerce({"detailed": "true"}, WEATHER)
        self.assertIs(value["detailed"], True)
        self.assertTrue(notes)

    def test_float_to_integer_when_whole(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        value, _ = coerce({"x": 4.0}, schema)
        self.assertEqual(value["x"], 4)
        self.assertIsInstance(value["x"], int)

    def test_float_not_truncated_when_fractional(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        value, _ = coerce({"x": 4.7}, schema)
        self.assertEqual(value["x"], 4.7)

    def test_enum_case_fix(self):
        value, notes = coerce({"units": "Metric"}, WEATHER)
        self.assertEqual(value["units"], "metric")
        self.assertTrue(any("enum" in n for n in notes))

    def test_number_to_string(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        value, notes = coerce({"x": 12}, schema)
        self.assertEqual(value["x"], "12")
        self.assertTrue(notes)

    def test_json_string_to_object(self):
        schema = {
            "type": "object",
            "properties": {
                "who": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
        }
        value, notes = coerce({"who": '{"name": "Ana"}'}, schema)
        self.assertEqual(value["who"], {"name": "Ana"})
        self.assertTrue(notes)

    def test_nested_coercion_recurses(self):
        schema = {
            "type": "object",
            "properties": {
                "traveller": {
                    "type": "object",
                    "properties": {"age": {"type": "integer"}},
                }
            },
        }
        value, notes = coerce({"traveller": {"age": "34"}}, schema)
        self.assertEqual(value["traveller"]["age"], 34)
        self.assertTrue(any("traveller.age" in n for n in notes))

    def test_array_items_coerced(self):
        schema = {"type": "object", "properties": {"ns": {"type": "array", "items": {"type": "integer"}}}}
        value, notes = coerce({"ns": ["1", "2"]}, schema)
        self.assertEqual(value["ns"], [1, 2])
        self.assertTrue(notes)

    def test_scalar_wrapped_into_array(self):
        schema = {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}}}}
        value, notes = coerce({"tags": "solo"}, schema)
        self.assertEqual(value["tags"], ["solo"])
        self.assertTrue(any("wrapped" in n for n in notes))

    def test_default_filled_for_missing_required(self):
        schema = {
            "type": "object",
            "required": ["units"],
            "properties": {"units": {"type": "string", "default": "metric"}},
        }
        value, notes = coerce({}, schema)
        self.assertEqual(value["units"], "metric")
        self.assertTrue(notes)

    def test_null_filled_from_default(self):
        schema = {"type": "object", "properties": {"units": {"type": "string", "default": "metric"}}}
        value, notes = coerce({"units": None}, schema)
        self.assertEqual(value["units"], "metric")

    def test_unknown_keys_preserved(self):
        value, _ = coerce({"city": "Paris", "extra": 1}, WEATHER)
        self.assertEqual(value["extra"], 1)

    def test_valid_input_unchanged(self):
        original = {"city": "Paris", "units": "metric", "days": 3}
        value, notes = coerce(dict(original), WEATHER)
        self.assertEqual(value, original)
        self.assertEqual(notes, [])

    def test_no_schema_is_noop(self):
        value, notes = coerce({"a": "1"}, {})
        self.assertEqual(value, {"a": "1"})
        self.assertEqual(notes, [])


class TestBareValueWhereObjectExpected(unittest.TestCase):
    """`questions: ["why?"]` means `questions: [{question: "why?"}]`.

    Observed from a real harness: gpt-oss sent an array of strings where the
    schema wanted an array of objects, and the harness rejected it with
    "questions[0] must be an object".
    """

    QUESTIONS = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "question"],
                    "properties": {
                        "id": {"type": "string"},
                        "question": {"type": "string"},
                        "header": {"type": "string"},
                    },
                },
            }
        },
    }

    def test_bare_string_is_wrapped_under_the_content_property(self):
        out, notes = coerce({"questions": ["Why is the sky blue?"]}, self.QUESTIONS)
        self.assertEqual(out["questions"][0]["question"], "Why is the sky blue?")
        self.assertTrue(any("wrapped bare value" in n for n in notes))

    def test_wrapping_then_fills_the_missing_id_so_it_validates(self):
        out, _ = coerce({"questions": ["Why?"]}, self.QUESTIONS)
        self.assertEqual(validate(out, self.QUESTIONS), [])

    def test_a_scalar_is_not_guessed_into_an_object_without_a_content_field(self):
        opaque = {
            "type": "object",
            "required": ["alpha", "beta"],
            "properties": {"alpha": {"type": "string"}, "beta": {"type": "string"}},
        }
        out, notes = coerce("stray", opaque)
        self.assertEqual(out, "stray")
        self.assertEqual(notes, [])

    def test_a_single_required_field_is_unambiguous_and_does_wrap(self):
        # With one field the value can only be that field, so this is not a guess.
        single = {
            "type": "object",
            "required": ["body"],
            "properties": {"body": {"type": "string"}},
        }
        out, notes = coerce("hello", single)
        self.assertEqual(out, {"body": "hello"})
        self.assertTrue(any("wrapped bare value" in n for n in notes))


class TestMintingMissingIdentifiers(unittest.TestCase):
    """An omitted identifier is filled only when that is safe.

    The danger is inventing a value that addresses something real, so minting
    requires an id-like name, no schema constraint, and every other required
    property already supplied.
    """

    ASK = {
        "type": "object",
        "required": ["id", "question"],
        "properties": {"id": {"type": "string"}, "question": {"type": "string"}},
    }

    def test_id_is_minted_when_the_substance_is_already_present(self):
        out, notes = coerce({"question": "Which mode?"}, self.ASK)
        self.assertTrue(out["id"])
        self.assertTrue(any("minted placeholder" in n for n in notes))
        self.assertEqual(validate(out, self.ASK), [])

    def test_minting_is_deterministic(self):
        first, _ = coerce({"question": "Which mode?"}, self.ASK)
        second, _ = coerce({"question": "Which mode?"}, self.ASK)
        self.assertEqual(first["id"], second["id"])

    def test_sibling_items_get_distinct_ids(self):
        schema = {
            "type": "object",
            "properties": {"questions": {"type": "array", "items": self.ASK}},
        }
        out, _ = coerce({"questions": [{"question": "a"}, {"question": "b"}]}, schema)
        ids = [q["id"] for q in out["questions"]]
        self.assertEqual(len(set(ids)), 2, ids)
        self.assertEqual(validate(out, schema), [])

    def test_an_id_that_is_the_only_requirement_is_not_minted(self):
        # Nothing else was supplied: minting here would invent a target.
        only_id = {
            "type": "object",
            "required": ["id"],
            "properties": {"id": {"type": "string"}},
        }
        out, notes = coerce({}, only_id)
        self.assertNotIn("id", out)
        self.assertEqual(notes, [])

    def test_content_fields_are_never_minted(self):
        # `question` is the substance; inventing it would fabricate the request.
        out, notes = coerce({"id": "q0"}, self.ASK)
        self.assertNotIn("question", out)
        self.assertEqual(notes, [])

    def test_a_patterned_id_is_not_minted(self):
        patterned = {
            "type": "object",
            "required": ["id", "question"],
            "properties": {
                "id": {"type": "string", "pattern": "^[0-9]+$"},
                "question": {"type": "string"},
            },
        }
        out, _ = coerce({"question": "Which mode?"}, patterned)
        self.assertNotIn("id", out)

    def test_a_default_still_wins_over_minting(self):
        defaulted = {
            "type": "object",
            "required": ["id", "question"],
            "properties": {
                "id": {"type": "string", "default": "fixed"},
                "question": {"type": "string"},
            },
        }
        out, notes = coerce({"question": "Which mode?"}, defaulted)
        self.assertEqual(out["id"], "fixed")
        self.assertTrue(any("schema default" in n for n in notes))

    def test_a_non_identifier_required_field_is_not_minted(self):
        # `header` is not an identifier, so its absence is the model's business.
        schema = {
            "type": "object",
            "required": ["id", "header"],
            "properties": {"id": {"type": "string"}, "header": {"type": "string"}},
        }
        out, _ = coerce({"id": "q0"}, schema)
        self.assertNotIn("header", out)


class TestSchemaHelpers(unittest.TestCase):
    def test_example_for_object(self):
        example = example_for(WEATHER)
        self.assertIsInstance(example, dict)
        self.assertIn("city", example)

    def test_example_uses_enum_first_value(self):
        schema = {"type": "object", "properties": {"units": {"type": "string", "enum": ["metric", "imperial"]}}}
        self.assertEqual(example_for(schema)["units"], "metric")

    def test_render_schema_hint_marks_required(self):
        rendered = render_schema_hint("get_weather", WEATHER)
        self.assertIn("get_weather", rendered)
        self.assertIn("required", rendered)
        self.assertIn("metric", rendered)

    def test_render_schema_hint_without_properties(self):
        self.assertIn("no arguments", render_schema_hint("noop", {"type": "object"}))


if __name__ == "__main__":
    unittest.main()
