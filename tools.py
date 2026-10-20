from langchain_core.tools import tool
from database import get_order, issue_refund


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order using its order ID."""
    return get_order(order_id) or {'error': 'Order not found'}


@tool
def process_refund(order_id: str, request_id: str) -> dict:
    """Record an approved refund in the demo ledger, idempotently by order ID."""
    return issue_refund(order_id, request_id)
