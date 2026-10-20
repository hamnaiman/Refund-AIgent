from datetime import date
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt
from agents import classify, compose_reply
from tools import lookup_order, process_refund


class SupportState(TypedDict, total=False):
    message: str
    request_id: str
    intent: str
    order_id: Optional[str]
    order: Optional[dict]
    decision: str
    detail: str
    approved: Optional[bool]
    refund_result: Optional[dict]
    response: str


def classify_node(state: SupportState):
    return classify(state['message'])


def route_intent(state: SupportState):
    return 'lookup' if state['intent'] in ('refund', 'tracking', 'cancellation', 'complaint') else 'reply'


def lookup_node(state: SupportState):
    if not state.get('order_id'):
        return {'order': None, 'decision': 'missing_order', 'detail': 'Please provide an order ID such as ORD-1001.'}
    order = lookup_order.invoke({'order_id': state['order_id']})
    if 'error' in order:
        return {'order': None, 'decision': 'not_found', 'detail': 'That order ID was not found.'}
    return {'order': order}


def route_lookup(state: SupportState):
    if not state.get('order'):
        return 'reply'
    return 'policy' if state['intent'] == 'refund' else 'non_refund'


def non_refund_node(state: SupportState):
    order = state['order']
    intent = state['intent']
    if intent == 'tracking':
        detail = f"Order {order['order_id']} has status {order['status']}; delivery date: {order['delivery_date']}."
    elif intent == 'cancellation':
        detail = 'Cancellation requests require a support agent. No cancellation was performed.'
    else:
        detail = 'Your complaint has been recorded for support review. No order changes were made.'
    return {'decision': 'information_only', 'detail': detail}


def policy_node(state: SupportState):
    order = state['order']
    if order['refund_status'] != 'none':
        return {'decision': 'rejected', 'detail': 'This order has already been refunded.'}
    if order['status'] != 'delivered' or not order['delivery_date']:
        return {'decision': 'rejected', 'detail': 'Only delivered orders are eligible under this demo policy.'}
    age = (date.today() - date.fromisoformat(order['delivery_date'])).days
    if age < 0 or age > 7:
        return {'decision': 'rejected', 'detail': f'Delivery was {age} days ago; the refund window is seven days.'}
    if order['amount'] > 100:
        return {'decision': 'review', 'detail': f"Refund of ${order['amount']:.2f} requires human approval."}
    return {'decision': 'approved', 'detail': 'Refund qualifies under the demo policy.'}


def route_policy(state: SupportState):
    if state['decision'] == 'review':
        return 'human_review'
    if state['decision'] == 'approved':
        return 'refund'
    return 'reply'


def human_review_node(state: SupportState):
    answer = interrupt({'order_id': state['order_id'], 'amount': state['order']['amount'],
                        'message': state['message'], 'question': 'Approve this refund?'})
    approved = answer.get('approved') is True if isinstance(answer, dict) else False
    return {'approved': approved, 'decision': 'approved' if approved else 'rejected',
            'detail': 'Human reviewer approved.' if approved else 'Human reviewer rejected.'}


def route_review(state: SupportState):
    return 'refund' if state['approved'] else 'reply'


def refund_node(state: SupportState):
    # Recheck eligibility after a potentially long human-review pause.
    fresh = lookup_order.invoke({'order_id': state['order_id']})
    if 'error' in fresh or fresh['refund_status'] != 'none':
        return {'decision': 'failed', 'detail': 'Order missing or already refunded.'}
    if fresh['status'] != 'delivered' or not fresh['delivery_date']:
        return {'decision': 'failed', 'detail': 'Order is no longer eligible.'}
    age = (date.today() - date.fromisoformat(fresh['delivery_date'])).days
    if age < 0 or age > 7:
        return {'decision': 'failed', 'detail': 'Refund window expired before processing.'}
    if fresh['amount'] > 100 and state.get('approved') is not True:
        return {'decision': 'failed', 'detail': 'Human approval is required.'}
    result = process_refund.invoke({'order_id': state['order_id'], 'request_id': state['request_id']})
    return {'refund_result': result, 'decision': 'recorded' if result['success'] else 'failed',
            'detail': result.get('note', result.get('reason', 'Refund operation failed'))}


def reply_node(state: SupportState):
    detail = state.get('detail', 'Answer the general question without inventing order details.')
    return {'response': compose_reply(state['message'], state.get('decision', 'general'), detail)}


def build_graph(checkpointer):
    builder = StateGraph(SupportState)
    builder.add_node('classify', classify_node)
    builder.add_node('lookup', lookup_node)
    builder.add_node('policy', policy_node)
    builder.add_node('human_review', human_review_node)
    builder.add_node('refund', refund_node)
    builder.add_node('non_refund', non_refund_node)
    builder.add_node('reply', reply_node)
    builder.add_edge(START, 'classify')
    builder.add_conditional_edges('classify', route_intent)
    builder.add_conditional_edges('lookup', route_lookup)
    builder.add_conditional_edges('policy', route_policy)
    builder.add_conditional_edges('human_review', route_review)
    builder.add_edge('non_refund', 'reply')
    builder.add_edge('refund', 'reply')
    builder.add_edge('reply', END)
    return builder.compile(checkpointer=checkpointer)
