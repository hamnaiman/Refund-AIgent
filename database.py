"""CSV-backed demo order and refund storage (single-process use only)."""
import csv
import os
import threading
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / 'data'
ORDERS = DATA_DIR / 'orders.csv'
REFUNDS = DATA_DIR / 'refunds.csv'
ORDER_FIELDS = ['order_id', 'customer_id', 'product', 'amount', 'status', 'delivery_date', 'refund_status']
REFUND_FIELDS = ['order_id', 'request_id', 'amount', 'created_at']
_LOCK = threading.RLock()


def _read(path):
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _write(path, rows, fields):
    """Replace CSV via a temporary file to reduce risk of partial writes."""
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def seed_db():
    """Create demo data on first launch; do not overwrite subsequent changes."""
    with _LOCK:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ORDERS.exists():
            today = date.today()
            rows = []
            for i in range(1, 21):
                amount = [75, 120, 350, 40, 95][(i - 1) % 5]
                days = 15 if i == 4 else 1 if i in (2, 3) else 3
                refunded = i == 5
                rows.append({
                    'order_id': f'ORD-{1000+i}', 'customer_id': f'CUST-{100+i}',
                    'product': 'Wireless Headphones', 'amount': amount,
                    'status': 'refunded' if refunded else 'delivered',
                    'delivery_date': (today - timedelta(days=days)).isoformat(),
                    'refund_status': 'refunded' if refunded else 'none',
                })
            _write(ORDERS, rows, ORDER_FIELDS)
        if not REFUNDS.exists():
            _write(REFUNDS, [], REFUND_FIELDS)


def _normalize(row):
    if row is None:
        return None
    return {**row, 'amount': float(row['amount'])}


def get_order(order_id: str):
    with _LOCK:
        for row in _read(ORDERS):
            if row['order_id'] == order_id:
                return _normalize(row)
    return None


def list_orders():
    with _LOCK:
        return [_normalize(row) for row in _read(ORDERS)]


def list_refunds():
    with _LOCK:
        return [{**row, 'amount': float(row['amount'])} for row in _read(REFUNDS)]


def issue_refund(order_id: str, request_id: str):
    """Demo only: idempotent in one Python process, not a payment transaction."""
    with _LOCK:
        orders = _read(ORDERS)
        order = next((r for r in orders if r['order_id'] == order_id), None)
        if order is None:
            return {'success': False, 'reason': 'Order not found'}
        if order['refund_status'] != 'none' or any(r['order_id'] == order_id for r in _read(REFUNDS)):
            return {'success': False, 'reason': 'Order already refunded'}
        if order['status'] != 'delivered':
            return {'success': False, 'reason': 'Order is not delivered'}
        amount = float(order['amount'])
        # Record refund first; subsequent attempts are blocked by refund history.
        refunds = _read(REFUNDS)
        refunds.append({'order_id': order_id, 'request_id': request_id,
                        'amount': amount, 'created_at': datetime.now(timezone.utc).isoformat()})
        _write(REFUNDS, refunds, REFUND_FIELDS)
        order['refund_status'] = 'refunded'
        order['status'] = 'refunded'
        _write(ORDERS, orders, ORDER_FIELDS)
        return {'success': True, 'amount': amount, 'order_id': order_id,
                'note': 'Recorded in CSV demo ledger; no real payment was sent'}
