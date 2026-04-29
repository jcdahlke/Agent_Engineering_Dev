"""
Input guardrail that validates research topics before any LLM work begins.

OpenAI Agents SDK patterns demonstrated:
  - @input_guardrail decorator: wraps an async function into an InputGuardrail
  - GuardrailFunctionOutput: tripwire_triggered=True aborts the run immediately
  - output_type=TopicValidation: forces the validator agent to return structured JSON
  - result.final_output_as(TopicValidation): type-safe extraction of structured output
  - InputGuardrailTripwireTriggered: the exception raised when tripwire fires
"""
from __future__ import annotations

from pydantic import BaseModel

from agents import Agent, GuardrailFunctionOutput, Runner, RunContextWrapper, input_guardrail

from config import settings
from context import ResearchContext


class TopicValidation(BaseModel):
    """Structured output produced by the guardrail validator agent."""
    is_valid: bool
    reason: str
    suggested_alternative: str = ""   # populated when is_valid=False


# A lightweight validator agent — uses a cheap model and returns structured JSON.
# It is NOT part of the main pipeline; it runs only on the initial user input.
_validator_agent = Agent(
    name="TopicValidator",
    model=settings.guardrail_model,
    instructions=(
        "You are a content safety validator. Assess the research topic provided.\n\n"
        "Return is_valid=True if the topic is a legitimate research subject.\n"
        "Return is_valid=False if the topic:\n"
        "  - Requests harmful, illegal, or dangerous information\n"
        "  - Is designed to generate misinformation or propaganda\n"
        "  - Is nonsensical or cannot be meaningfully researched\n\n"
        "Always provide a clear reason for your decision."
    ),
    output_type=TopicValidation,
)


@input_guardrail
async def topic_safety_guardrail(
    ctx: RunContextWrapper[ResearchContext],
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:
    """
    Runs before the Supervisor agent starts.
    Sets tripwire_triggered=True to abort the pipeline if the topic is unsafe.
    """
    result = await Runner.run(
        _validator_agent,
        input,
        context=ctx.context,
    )
    validation = result.final_output_as(TopicValidation)

    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=not validation.is_valid,
    )
