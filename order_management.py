"""CrewAI order-support workflow backed by local CSV demo data."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool
from litellm.exceptions import APIError
from openai import APIError as OpenAIAPIError

DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
ORDERS_FILE = DATA_DIRECTORY / "order_management_orders.csv"
INVENTORY_FILE = DATA_DIRECTORY / "order_management_inventory.csv"
VALID_SHIPMENT_STATUSES = {"processing", "in_transit", "delayed", "delivered", "lost", "exception"}
HUMAN_REQUEST_TERMS = ("human", "person", "representative", "manager", "supervisor", "complaint")
NOT_RECEIVED_TERMS = ("not received", "never arrived", "didn't arrive", "did not arrive", "missing package")


class OrderManagementError(ValueError):
    """Raised when an order request, demo data, or LLM operation cannot be handled."""


@dataclass(frozen=True)
class OrderRecord:
    order_id: str
    customer_name: str
    product_id: str
    product_name: str
    quantity: int
    order_date: date
    expected_delivery: date
    revised_delivery: date | None
    shipment_status: str
    tracking_id: str
    current_location: str
    tracking_note: str
    issue_count: int


@dataclass(frozen=True)
class InventoryRecord:
    product_id: str
    product_name: str
    stock: int
    alternatives: tuple[str, ...]


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise OrderManagementError(f"The demo data file {path.name} has no header.")
            return list(reader)
    except OSError as exc:
        raise OrderManagementError(f"Could not read demo data file {path.name}: {exc}") from exc


def _parse_date(value: str, field: str, order_id: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise OrderManagementError(
            f"Order {order_id} has an invalid {field} date in the demo data."
        ) from exc


def find_order(order_id: str) -> OrderRecord | None:
    """Find an order from the local CSV sample."""
    normalized_id = order_id.strip().upper()
    if not normalized_id:
        raise OrderManagementError("Enter an order ID.")
    for row in _read_csv(ORDERS_FILE):
        row_id = row.get("order_id", "").strip().upper()
        if row_id != normalized_id:
            continue
        try:
            quantity = int(row["quantity"])
            issue_count = int(row["issue_count"])
            status = row["shipment_status"].strip().lower()
            if quantity < 1 or issue_count < 0:
                raise ValueError("quantity and issue count must be non-negative.")
            if status not in VALID_SHIPMENT_STATUSES:
                raise ValueError(f"unsupported shipment status {status!r}.")
            revised_value = row.get("revised_delivery", "").strip()
            revised_delivery = (
                _parse_date(revised_value, "revised delivery", normalized_id)
                if revised_value
                else None
            )
            return OrderRecord(
                order_id=normalized_id,
                customer_name=row["customer_name"].strip(),
                product_id=row["product_id"].strip().upper(),
                product_name=row["product_name"].strip(),
                quantity=quantity,
                order_date=_parse_date(row["order_date"], "order", normalized_id),
                expected_delivery=_parse_date(
                    row["expected_delivery"], "expected delivery", normalized_id
                ),
                revised_delivery=revised_delivery,
                shipment_status=status,
                tracking_id=row["tracking_id"].strip(),
                current_location=row["current_location"].strip(),
                tracking_note=row["tracking_note"].strip(),
                issue_count=issue_count,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OrderManagementError(
                f"Order {normalized_id} has invalid fields in the demo data: {exc}"
            ) from exc
    return None


def find_inventory(product_id: str) -> InventoryRecord | None:
    """Find current stock and alternative product IDs from the local CSV."""
    normalized_id = product_id.strip().upper()
    if not normalized_id:
        raise OrderManagementError("An order product ID is required for inventory lookup.")
    for row in _read_csv(INVENTORY_FILE):
        if row.get("product_id", "").strip().upper() != normalized_id:
            continue
        try:
            stock = int(row["stock"])
            if stock < 0:
                raise ValueError("stock cannot be negative.")
            alternatives = tuple(
                item.strip().upper()
                for item in row.get("alternative_product_ids", "").split("|")
                if item.strip()
            )
            return InventoryRecord(
                product_id=normalized_id,
                product_name=row["product_name"].strip(),
                stock=stock,
                alternatives=alternatives,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise OrderManagementError(
                f"Inventory record {normalized_id} is invalid: {exc}"
            ) from exc
    return None


def get_inventory_evidence(product_id: str) -> dict[str, Any]:
    """Return stock and only currently available alternatives."""
    inventory = find_inventory(product_id)
    if inventory is None:
        return {"found": False, "product_id": product_id, "error": "Product not found."}
    alternatives: list[dict[str, Any]] = []
    for alternative_id in inventory.alternatives:
        alternative = find_inventory(alternative_id)
        if alternative is not None and alternative.stock > 0:
            alternatives.append(
                {
                    "product_id": alternative.product_id,
                    "product_name": alternative.product_name,
                    "stock": alternative.stock,
                }
            )
    return {
        "found": True,
        "product_id": inventory.product_id,
        "product_name": inventory.product_name,
        "stock": inventory.stock,
        "available_alternatives": alternatives,
    }


def get_tracking_evidence(order: OrderRecord | None) -> dict[str, Any]:
    """Return the shipment facts recorded for an order."""
    if order is None:
        return {"found": False, "error": "No tracking record is available for this order."}
    return {
        "found": True,
        "tracking_id": order.tracking_id,
        "status": order.shipment_status,
        "expected_delivery": order.expected_delivery.isoformat(),
        "revised_delivery": (
            order.revised_delivery.isoformat() if order.revised_delivery else None
        ),
        "current_location": order.current_location,
        "tracking_note": order.tracking_note,
    }


def order_lookup_tool(order_id: str) -> str:
    """Look up a sample order using its order ID."""
    order = find_order(order_id)
    if order is None:
        return json.dumps({"found": False, "order_id": order_id, "error": "Order not found."})
    return json.dumps({"found": True, **_order_to_dict(order)}, ensure_ascii=False)


def inventory_lookup_tool(product_id: str) -> str:
    """Check product stock and list alternatives that are in stock."""
    return json.dumps(get_inventory_evidence(product_id), ensure_ascii=False)


def shipment_tracking_tool(order_id: str) -> str:
    """Look up the sample tracking status for an order ID."""
    return json.dumps(get_tracking_evidence(find_order(order_id)), ensure_ascii=False)


order_lookup = tool("Order lookup")(order_lookup_tool)
inventory_lookup = tool("Inventory lookup")(inventory_lookup_tool)
shipment_tracking = tool("Shipment tracking")(shipment_tracking_tool)


def _order_to_dict(order: OrderRecord) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "customer_name": order.customer_name,
        "product_id": order.product_id,
        "product_name": order.product_name,
        "quantity": order.quantity,
        "order_date": order.order_date.isoformat(),
        "expected_delivery": order.expected_delivery.isoformat(),
        "revised_delivery": (
            order.revised_delivery.isoformat() if order.revised_delivery else None
        ),
        "shipment_status": order.shipment_status,
        "tracking_id": order.tracking_id,
        "current_location": order.current_location,
        "tracking_note": order.tracking_note,
        "issue_count": order.issue_count,
    }


def requires_human_escalation(
    order: OrderRecord | None,
    inventory: dict[str, Any],
    tracking: dict[str, Any],
    question: str,
) -> tuple[bool, list[str]]:
    """Apply explicit escalation safeguards to the resolution agent's recommendation."""
    reasons: list[str] = []
    normalized_question = question.casefold()

    if order is None:
        reasons.append("The order could not be verified.")
    elif not inventory.get("found"):
        reasons.append("The ordered product could not be verified in the inventory system.")
    if any(term in normalized_question for term in HUMAN_REQUEST_TERMS):
        reasons.append("The customer asked to speak with a person.")
    if order is not None and order.issue_count >= 2:
        reasons.append("This order has two or more prior support issues.")
    if tracking.get("status") in {"lost", "exception"}:
        reasons.append("The carrier marked the shipment lost or reported an exception.")
    if (
        tracking.get("status") == "delivered"
        and any(term in normalized_question for term in NOT_RECEIVED_TERMS)
    ):
        reasons.append("The shipment is marked delivered, but the customer says it is missing.")
    if (
        inventory.get("found")
        and inventory.get("stock") == 0
        and not inventory.get("available_alternatives")
    ):
        reasons.append("The ordered item is out of stock and there is no available alternative.")
    return bool(reasons), reasons


def _create_llm(provider: str, model: str | None = None) -> LLM:
    provider = provider.strip().lower()
    if provider == "ollama":
        selected_model = model or os.getenv("ORDER_OLLAMA_MODEL", "ollama/llama3.2")
        return LLM(
            model=selected_model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.2,
        )
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise OrderManagementError("Set OPENAI_API_KEY to use the OpenAI provider.")
        return LLM(
            model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=api_key,
            temperature=0.2,
        )
    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise OrderManagementError(
                "Set OPENROUTER_API_KEY to use the OpenRouter provider."
            )
        return LLM(
            model=model or os.getenv("OPENROUTER_MODEL", "openrouter/openai/gpt-4o-mini"),
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2,
        )
    raise OrderManagementError("Choose Ollama, OpenAI, or OpenRouter as the LLM provider.")


def _build_crew(
    order_id: str,
    question: str,
    provider: str,
    model: str | None,
    order: OrderRecord | None,
    inventory: dict[str, Any],
    tracking: dict[str, Any],
    policy_required: bool,
    escalation_reasons: list[str],
) -> Crew:
    llm = _create_llm(provider, model)
    support_agent = Agent(
        role="Customer Support Agent",
        goal="Identify the customer's order and issue, and route it using verified order facts.",
        backstory="You are a careful support triage specialist. You never invent order information.",
        tools=[order_lookup],
        llm=llm,
        verbose=False,
    )
    inventory_agent = Agent(
        role="Inventory Agent",
        goal="Verify item availability and identify only in-stock replacement alternatives.",
        backstory="You rely on the inventory system and never promise an unavailable item.",
        tools=[inventory_lookup],
        llm=llm,
        verbose=False,
    )
    delivery_agent = Agent(
        role="Delivery Agent",
        goal="Verify carrier status, delivery estimate, and whether the order is delayed.",
        backstory="You use tracking evidence only and never invent carrier events or dates.",
        tools=[shipment_tracking],
        llm=llm,
        verbose=False,
    )
    resolution_agent = Agent(
        role="Resolution Agent",
        goal="Combine verified support, inventory, and delivery findings into a clear response and decide whether a human must take over.",
        backstory=(
            "You prioritize customer safety and accurate commitments. You never claim that "
            "a refund, replacement, or order change has already been made."
        ),
        llm=llm,
        verbose=False,
    )

    order_facts = (
        json.dumps(_order_to_dict(order), ensure_ascii=False)
        if order is not None
        else json.dumps({"found": False, "order_id": order_id, "error": "Order not found."})
    )
    task_support = Task(
        description=(
            "Triage the customer's issue. Verify the supplied order ID using the order lookup tool "
            "and summarize the verified order facts; distinguish the customer's claim from system "
            "facts. Do not promise a remedy.\n"
            f"Order ID: {order_id}\nCustomer question: {question}\n"
            f"Verified order record: {order_facts}"
        ),
        expected_output="Order verification, product, shipment status, and concise issue classification.",
        agent=support_agent,
    )
    task_inventory = Task(
        description=(
            "Check the ordered product and possible replacements using the inventory lookup tool. "
            "Only describe available alternatives as options; no replacement is authorized.\n"
            f"Order ID: {order_id}\nVerified order record: {order_facts}\n"
            f"Verified inventory evidence: {json.dumps(inventory, ensure_ascii=False)}"
        ),
        expected_output="Current stock count and available alternatives, or a clear unavailable result.",
        agent=inventory_agent,
        context=[task_support],
    )
    task_delivery = Task(
        description=(
            "Check shipment tracking using the shipment tracking tool. Explain the verified status, "
            "location, expected date, and revised estimate if present. Do not invent an ETA.\n"
            f"Order ID: {order_id}\nVerified tracking evidence: {json.dumps(tracking, ensure_ascii=False)}"
        ),
        expected_output="Verified tracking status, location, delivery estimate, and delay finding.",
        agent=delivery_agent,
        context=[task_support],
    )
    task_resolution = Task(
        description=(
            "Use the support, inventory, and delivery task outputs plus the verified evidence below "
            "to draft a concise customer response. First line MUST be exactly "
            "'ESCALATION: YES' or 'ESCALATION: NO'. Then provide the customer response and one "
            "sentence explaining the escalation decision. The resolution policy requires escalation "
            "when any mandatory reason is listed below; never override a mandatory escalation. "
            "Do not claim that a refund, replacement, or order change has already occurred.\n"
            f"Mandatory policy escalation: {policy_required}\n"
            f"Mandatory escalation reasons: {json.dumps(escalation_reasons, ensure_ascii=False)}\n"
            f"Order facts: {order_facts}\n"
            f"Inventory evidence: {json.dumps(inventory, ensure_ascii=False)}\n"
            f"Tracking evidence: {json.dumps(tracking, ensure_ascii=False)}\n"
            f"Customer question: {question}"
        ),
        expected_output=(
            "First line 'ESCALATION: YES' or 'ESCALATION: NO', followed by an accurate "
            "customer response and escalation rationale."
        ),
        agent=resolution_agent,
        context=[task_support, task_inventory, task_delivery],
    )
    return Crew(
        agents=[support_agent, inventory_agent, delivery_agent, resolution_agent],
        tasks=[task_support, task_inventory, task_delivery, task_resolution],
        process=Process.sequential,
        verbose=False,
    )


def _kickoff(crew: Crew) -> str:
    try:
        return str(crew.kickoff())
    except (APIError, OpenAIAPIError, ConnectionError, TimeoutError) as exc:
        raise OrderManagementError(
            f"The selected LLM could not complete the request ({type(exc).__name__}). "
            "Check that Ollama is running and the model is installed, or verify the "
            "selected provider's API key, model name, and network connection."
        ) from exc


def _agent_escalated(response: str) -> bool | None:
    match = re.match(r"^\s*ESCALATION:\s*(YES|NO)\b", response, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).upper() == "YES"


def _customer_response(response: str) -> str:
    """Hide the machine-readable escalation marker from the customer-facing reply."""
    lines = response.splitlines()
    if lines and re.match(r"^\s*ESCALATION:\s*(YES|NO)\b", lines[0], flags=re.IGNORECASE):
        return "\n".join(lines[1:]).strip()
    return response.strip()


def handle_order_query(
    order_id: str,
    question: str,
    provider: str = "ollama",
    model: str | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Run the four-agent workflow and enforce mandatory human-escalation rules."""
    order_id = order_id.strip().upper()
    question = question.strip()
    if not order_id:
        raise OrderManagementError("Enter an order ID.")
    if not question:
        raise OrderManagementError("Enter a question about the order.")

    order = find_order(order_id)
    inventory = get_inventory_evidence(order.product_id) if order else {
        "found": False,
        "error": "Cannot check inventory until the order is verified.",
    }
    tracking = get_tracking_evidence(order)
    escalation_required, escalation_reasons = requires_human_escalation(
        order, inventory, tracking, question
    )
    crew = _build_crew(
        order_id,
        question,
        provider,
        model,
        order,
        inventory,
        tracking,
        escalation_required,
        escalation_reasons,
    )
    response = _kickoff(crew)
    agent_escalation = _agent_escalated(response)
    escalate = escalation_required or agent_escalation is True or agent_escalation is None
    if agent_escalation is None:
        escalation_reasons = [
            *escalation_reasons,
            "The resolution agent did not return the required escalation decision format.",
        ]
    elif agent_escalation and not escalation_required:
        escalation_reasons = ["The resolution agent requested human review."]

    return {
        "response": _customer_response(response),
        "resolution_agent_output": response,
        "order": _order_to_dict(order) if order else None,
        "inventory": inventory,
        "tracking": tracking,
        "escalation_required": escalate,
        "escalation_reasons": escalation_reasons,
        "llm_provider": provider,
        "llm_model": model or os.getenv("ORDER_OLLAMA_MODEL", "ollama/llama3.2"),
        "evaluated_on": (today or date.today()).isoformat(),
    }
