"""Structured Rule Engine — learned rules with confidence, scope, and lifecycle.

Rules are created from user corrections or pattern detection, applied during
SQL generation, confirmed/weakened based on query outcomes, and expired when
confidence drops below threshold.

Lifecycle:
  Created (0.9) → Applied → Confirmed (+0.05) or Weakened (-0.10) → Expired (<0.3)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger()

RULES_FILE = "rules.json"
CONFIDENCE_DECAY = 0.10
CONFIDENCE_BOOST = 0.05
EXPIRY_THRESHOLD = 0.30


def _rules_path(workspace_id: str) -> str:
    return os.path.join(
        os.path.expanduser("~"), ".sqlagent", "uploads",
        workspace_id, RULES_FILE,
    )


def load_rules(workspace_id: str, active_only: bool = True) -> list[dict]:
    """Load rules for a workspace. Returns active rules sorted by confidence * hit_count."""
    if not workspace_id:
        return []
    path = _rules_path(workspace_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            rules = json.load(f)
        if active_only:
            rules = [r for r in rules if r.get("confidence", 0) >= EXPIRY_THRESHOLD]
        # Sort: highest impact first (confidence * log(hit_count + 1))
        import math
        rules.sort(key=lambda r: r.get("confidence", 0) * math.log(r.get("hit_count", 0) + 2), reverse=True)
        return rules
    except Exception:
        return []


def save_rules(workspace_id: str, rules: list[dict]) -> None:
    """Persist rules to disk."""
    path = _rules_path(workspace_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(rules, f, indent=2)


def create_rule(
    workspace_id: str,
    text: str,
    scope: str = "workspace",
    source: str = "user_correction",
    confidence: float = 0.9,
    query_example: str = "",
) -> dict:
    """Create a new learned rule."""
    rules = load_rules(workspace_id, active_only=False)

    # Deduplicate: if a similar rule already exists, boost its confidence instead
    for existing in rules:
        if existing.get("text", "").lower().strip() == text.lower().strip():
            existing["confidence"] = min(existing["confidence"] + CONFIDENCE_BOOST, 0.99)
            existing["hit_count"] = existing.get("hit_count", 0) + 1
            existing["last_used_at"] = datetime.now(timezone.utc).isoformat()
            save_rules(workspace_id, rules)
            logger.info("rules.boosted", rule_id=existing["rule_id"], confidence=existing["confidence"])
            return existing

    rule = {
        "rule_id": f"rule_{uuid.uuid4().hex[:8]}",
        "text": text,
        "scope": scope,
        "source": source,
        "confidence": confidence,
        "hit_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_used_at": "",
        "query_example": query_example[:200],
    }
    rules.append(rule)
    save_rules(workspace_id, rules)
    logger.info("rules.created", rule_id=rule["rule_id"], text=text[:60], source=source)
    return rule


def record_rule_outcome(workspace_id: str, rule_ids: list[str], succeeded: bool) -> None:
    """Record whether rules led to success or failure. Adjusts confidence."""
    if not rule_ids or not workspace_id:
        return
    rules = load_rules(workspace_id, active_only=False)
    changed = False
    for rule in rules:
        if rule["rule_id"] in rule_ids:
            rule["hit_count"] = rule.get("hit_count", 0) + 1
            rule["last_used_at"] = datetime.now(timezone.utc).isoformat()
            if succeeded:
                rule["success_count"] = rule.get("success_count", 0) + 1
                rule["confidence"] = min(rule.get("confidence", 0.5) + CONFIDENCE_BOOST, 0.99)
            else:
                rule["confidence"] = max(rule.get("confidence", 0.5) - CONFIDENCE_DECAY, 0.0)
            rule["success_rate"] = (
                rule["success_count"] / rule["hit_count"]
                if rule["hit_count"] > 0 else 0.0
            )
            changed = True
    if changed:
        save_rules(workspace_id, rules)
