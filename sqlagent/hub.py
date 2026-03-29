"""QueryHub — community NL→SQL training packs.

Install pre-built training pairs for your industry. Accuracy improves immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from sqlagent.models import TrainingExample

logger = structlog.get_logger()


@dataclass
class HubPack:
    name: str
    description: str = ""
    domain: str = ""
    example_count: int = 0
    examples: list[TrainingExample] = field(default_factory=list)


# Bundled packs — these ship with sqlagent, no download needed
_BUNDLED_PACKS: dict[str, HubPack] = {
    "retail-asean": HubPack(
        name="retail-asean", domain="retail",
        description="10 verified NL→SQL pairs for ASEAN retail analytics",
        example_count=10,
        examples=[
            TrainingExample(nl_query="top 10 stores by revenue", sql="SELECT store_name, SUM(total_amount) AS revenue FROM orders o JOIN stores s ON o.store_id = s.store_id GROUP BY store_name ORDER BY revenue DESC LIMIT 10"),
            TrainingExample(nl_query="revenue by month this year", sql="SELECT strftime('%Y-%m', order_date) AS month, SUM(total_amount) AS revenue FROM orders WHERE strftime('%Y', order_date) = strftime('%Y', 'now') GROUP BY month ORDER BY month"),
            TrainingExample(nl_query="top customers by order count", sql="SELECT c.company_name, COUNT(*) AS order_count FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.company_name ORDER BY order_count DESC LIMIT 10"),
            TrainingExample(nl_query="average order value by country", sql="SELECT c.country, AVG(o.total_amount) AS avg_order FROM orders o JOIN customers c ON o.customer_id = c.customer_id GROUP BY c.country ORDER BY avg_order DESC"),
            TrainingExample(nl_query="products never ordered", sql="SELECT p.product_name FROM products p LEFT JOIN order_items oi ON p.product_id = oi.product_id WHERE oi.item_id IS NULL"),
            TrainingExample(nl_query="daily revenue trend", sql="SELECT order_date, SUM(total_amount) AS daily_revenue FROM orders GROUP BY order_date ORDER BY order_date"),
            TrainingExample(nl_query="top product categories by revenue", sql="SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi JOIN products p ON oi.product_id = p.product_id GROUP BY p.category ORDER BY revenue DESC"),
            TrainingExample(nl_query="customer retention rate", sql="SELECT COUNT(DISTINCT CASE WHEN order_count > 1 THEN customer_id END) * 100.0 / COUNT(DISTINCT customer_id) AS retention_pct FROM (SELECT customer_id, COUNT(*) AS order_count FROM orders GROUP BY customer_id)"),
            TrainingExample(nl_query="orders by day of week", sql="SELECT CASE strftime('%w', order_date) WHEN '0' THEN 'Sun' WHEN '1' THEN 'Mon' WHEN '2' THEN 'Tue' WHEN '3' THEN 'Wed' WHEN '4' THEN 'Thu' WHEN '5' THEN 'Fri' WHEN '6' THEN 'Sat' END AS day, COUNT(*) AS orders FROM orders GROUP BY day"),
            TrainingExample(nl_query="revenue per employee by store", sql="SELECT s.store_name, SUM(o.total_amount) AS revenue, COUNT(DISTINCT st.employee_id) AS staff, ROUND(SUM(o.total_amount) / COUNT(DISTINCT st.employee_id), 2) AS rpe FROM orders o JOIN stores s ON o.store_id = s.store_id JOIN staff st ON s.store_id = st.store_id GROUP BY s.store_name ORDER BY rpe DESC"),
        ],
    ),
    "ecommerce-standard": HubPack(
        name="ecommerce-standard", domain="ecommerce",
        description="5 verified NL→SQL pairs for e-commerce analytics",
        example_count=5,
        examples=[
            TrainingExample(nl_query="cart abandonment rate", sql="SELECT COUNT(CASE WHEN status = 'abandoned' THEN 1 END) * 100.0 / COUNT(*) AS abandonment_rate FROM carts"),
            TrainingExample(nl_query="top selling products this month", sql="SELECT p.name, SUM(oi.quantity) AS units_sold FROM order_items oi JOIN products p ON oi.product_id = p.id WHERE oi.created_at >= DATE_TRUNC('month', CURRENT_DATE) GROUP BY p.name ORDER BY units_sold DESC LIMIT 10"),
            TrainingExample(nl_query="average time to first purchase", sql="SELECT AVG(JULIANDAY(first_order) - JULIANDAY(signup_date)) AS avg_days FROM (SELECT user_id, MIN(created_at) AS first_order, u.created_at AS signup_date FROM orders o JOIN users u ON o.user_id = u.id GROUP BY user_id, u.created_at)"),
            TrainingExample(nl_query="revenue by traffic source", sql="SELECT utm_source, SUM(total) AS revenue, COUNT(*) AS orders FROM orders WHERE utm_source IS NOT NULL GROUP BY utm_source ORDER BY revenue DESC"),
            TrainingExample(nl_query="refund rate by category", sql="SELECT p.category, COUNT(CASE WHEN o.status = 'refunded' THEN 1 END) * 100.0 / COUNT(*) AS refund_rate FROM orders o JOIN order_items oi ON o.id = oi.order_id JOIN products p ON oi.product_id = p.id GROUP BY p.category ORDER BY refund_rate DESC"),
        ],
    ),
    "finance-standard": HubPack(
        name="finance-standard", domain="finance",
        description="4 verified NL→SQL pairs for financial analytics",
        example_count=4,
        examples=[
            TrainingExample(nl_query="monthly burn rate", sql="SELECT strftime('%Y-%m', date) AS month, SUM(CASE WHEN type = 'expense' THEN amount ELSE 0 END) AS burn FROM transactions GROUP BY month ORDER BY month DESC LIMIT 12"),
            TrainingExample(nl_query="runway in months", sql="SELECT CAST(SUM(CASE WHEN type = 'asset' THEN balance END) / (SUM(CASE WHEN type = 'expense' THEN amount END) / 12.0) AS INTEGER) AS runway_months FROM accounts a LEFT JOIN (SELECT type, SUM(amount) AS amount FROM transactions WHERE date >= DATE('now', '-12 months') GROUP BY type) t ON 1=1"),
            TrainingExample(nl_query="top expense categories", sql="SELECT category, SUM(amount) AS total FROM transactions WHERE type = 'expense' GROUP BY category ORDER BY total DESC LIMIT 10"),
            TrainingExample(nl_query="revenue growth rate", sql="SELECT curr.month, curr.revenue, ROUND((curr.revenue - prev.revenue) * 100.0 / prev.revenue, 1) AS growth_pct FROM (SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS revenue FROM transactions WHERE type = 'revenue' GROUP BY month) curr LEFT JOIN (SELECT strftime('%Y-%m', date) AS month, SUM(amount) AS revenue FROM transactions WHERE type = 'revenue' GROUP BY month) prev ON prev.month = strftime('%Y-%m', DATE(curr.month || '-01', '-1 month')) ORDER BY curr.month DESC"),
        ],
    ),
}


def list_packs() -> list[dict]:
    """List available QueryHub packs."""
    return [
        {
            "name": p.name,
            "description": p.description,
            "domain": p.domain,
            "example_count": p.example_count,
        }
        for p in _BUNDLED_PACKS.values()
    ]


async def install_pack(pack_name: str, example_store: Any) -> int:
    """Install a QueryHub pack into the example store. Returns count added."""
    pack = _BUNDLED_PACKS.get(pack_name)
    if not pack:
        raise ValueError(f"Pack '{pack_name}' not found. Available: {list(_BUNDLED_PACKS.keys())}")

    if not example_store:
        raise ValueError("No example store configured")

    count = await example_store.add_batch(pack.examples)
    logger.info("hub.installed", pack=pack_name, examples=count)
    return count
