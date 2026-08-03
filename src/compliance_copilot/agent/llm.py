import json
from collections.abc import Callable

from langfuse import observe
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from compliance_copilot.config import Settings


class StructuredOutputError(Exception):
    """Raised when an LLM response fails Pydantic (or business-rule) validation
    twice in a row. No silent fallback — the caller must handle this explicitly."""


def _client(settings: Settings) -> OpenAI:
    if settings.deepseek_api_key is None:
        raise StructuredOutputError("DEEPSEEK_API_KEY is not configured")
    return OpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
    )


@observe(as_type="generation")
def call_structured[T: BaseModel](
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    extra_validate: Callable[[T], None] | None = None,
) -> T:
    """Calls DeepSeek in JSON mode and validates the response against
    response_model. No free-text parsing, no regex over the output — the schema
    is handed to the model and Pydantic is the only thing that decides whether
    the response is acceptable.

    On a validation failure (malformed JSON, schema mismatch, or extra_validate
    raising), retries exactly once with the error fed back into the conversation.
    A second failure raises StructuredOutputError rather than returning something
    unvalidated — callers must handle that explicitly, not have it hidden from
    them.

    temperature=0: found via a live flaky test, not assumed — the same
    unambiguous "plush teddy bear, no electronics" spec occasionally reasoned
    its way to a different categorical conclusion across otherwise-identical
    calls at the default temperature. Wrong-but-consistent is debuggable;
    wrong-sometimes on identical input is a much worse property for a
    compliance system to have, so consistency is worth the tradeoff even
    though it doesn't fully eliminate provider-side sampling variance.
    """
    client = _client(settings)
    schema = response_model.model_json_schema()
    full_system_prompt = (
        f"{system_prompt}\n\n"
        "The following JSON Schema describes the REQUIRED SHAPE of your answer. "
        "Respond with a JSON object that is a valid INSTANCE of this schema — "
        "populated with actual values derived from the user message — NOT the "
        "schema definition itself, and not an object containing keys like "
        f"'properties' or 'type'.\n\nJSON Schema:\n{json.dumps(schema)}"
    )
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": full_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(2):
        if attempt == 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was invalid: {last_error}. "
                        "Respond again with valid JSON matching the schema exactly."
                    ),
                }
            )
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content or ""})
        try:
            if content is None:
                raise ValueError("empty response content")
            parsed = response_model.model_validate_json(content)
            if extra_validate is not None:
                extra_validate(parsed)
            return parsed
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            continue

    raise StructuredOutputError(
        f"LLM output failed validation twice for {response_model.__name__}: {last_error}"
    )
