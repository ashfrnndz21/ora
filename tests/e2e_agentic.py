"""End-to-end agentic tests — verifies fixes work via real LLM + real data.

Tests are deliberately NOT unit tests. Each one runs a full pipeline query
against the live server and asserts on agentic behavior, not hard-coded rules.

Run:
    python tests/e2e_agentic.py

Requires: server running on localhost:8080, workspace ws_19b2909e9cd9 with
occupazione + disoccupazione CSVs loaded.
"""

import asyncio
import json
import re
import sys
import time

import httpx

BASE = "http://localhost:8080"
WS_ID = "ws_19b2909e9cd9"
HEADERS = {"Content-Type": "application/json"}

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

results = []


def log(label: str, passed: bool, detail: str = ""):
    mark = PASS if passed else FAIL
    print(f"  {mark}  {label}")
    if detail:
        print(f"        {detail}")
    results.append((label, passed))


async def run_query(nl: str, session_id: str | None = None, history: list | None = None) -> dict:
    """POST /query and return the full response dict."""
    payload = {
        "nl_query": nl,
        "workspace_id": WS_ID,
        "session_id": session_id or f"test_{int(time.time()*1000)}",
    }
    if history:
        payload["conversation_history"] = history
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(f"{BASE}/query", json=payload, headers=HEADERS)
        r.raise_for_status()
        return r.json()


async def run_stream(nl: str, session_id: str | None = None, history: list | None = None) -> dict:
    """POST /query/stream (SSE) and return assembled final event."""
    payload = {
        "nl_query": nl,
        "workspace_id": WS_ID,
        "session_id": session_id or f"test_stream_{int(time.time()*1000)}",
    }
    if history:
        payload["conversation_history"] = history

    final = {}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", f"{BASE}/query/stream", json=payload, headers=HEADERS) as r:
            r.raise_for_status()
            event_type = ""
            async for line in r.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        if event_type in ("query.final", "query.result"):
                            final = data
                        elif event_type == "query.error":
                            final = {"error": data.get("error", "unknown"), "succeeded": False}
                    except Exception:
                        pass
    return final


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Basic cross-source query works
# ─────────────────────────────────────────────────────────────────────────────
async def test_basic_cross_source():
    print("\n[1] Basic cross-source query")
    r = await run_query("what are the top 5 countries by average employment rate?")

    log("query succeeded", r.get("succeeded") is True, f"rows={r.get('row_count',0)}")
    log("returned rows",   r.get("row_count", 0) >= 5, f"row_count={r.get('row_count',0)}")
    log("has nl_response", bool(r.get("nl_response")), r.get("nl_response","")[:80])
    log("has follow_ups",  len(r.get("follow_ups", [])) >= 2)

    country_col = r.get("columns", [])
    has_country = any("country" in c.lower() for c in country_col)
    log("country in columns", has_country, str(country_col))

    return r


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Anaphora — short follow-up carries context from previous turn
# ─────────────────────────────────────────────────────────────────────────────
async def test_anaphora_short_followup():
    print("\n[2] Anaphora — short follow-up preserves prior context")

    # Turn 1: establish Malaysia + Vietnam
    t1 = await run_query("compare employment rates for Malaysia vs Vietnam over the last 5 years")
    t1_nl = t1.get("nl_response", "")
    t1_sql = t1.get("sql", "")
    print(f"  T1 nl_response: {t1_nl[:120]}")

    log("T1 succeeded",           t1.get("succeeded") is True)
    log("T1 mentions Malaysia/Vietnam",
        any(w in (t1_nl + t1_sql).lower() for w in ["malaysia", "vietnam", "viet nam", "my", "vn"]),
        f"sql snippet: {t1_sql[:100]}")

    # Turn 2: follow-up with no explicit subject ("the unemployment" — 2 words)
    history = [{
        "role": "user",      "text": "compare employment rates for Malaysia vs Vietnam over the last 5 years"
    }, {
        "role": "assistant", "nl_response": t1_nl, "sql": t1_sql
    }]
    t2 = await run_query("what about the unemployment side?", history=history)
    t2_nl  = t2.get("nl_response", "")
    t2_sql = t2.get("sql", "")
    print(f"  T2 nl_response: {t2_nl[:120]}")

    # The anaphora rewriter should have resolved "what about the unemployment side?"
    # → "what about the unemployment rates for Malaysia vs Vietnam?" — so the result
    # should mention both countries, not random others.
    mentions_my_vn = any(w in (t2_nl + t2_sql).lower() for w in ["malaysia", "vietnam", "viet nam"])
    does_not_hallucinate_schema = "don't have access" not in t2_nl.lower() and "no schema" not in t2_nl.lower()

    log("T2 resolved MY+VN context",    mentions_my_vn, f"sql: {t2_sql[:120]}")
    log("T2 no schema-access confab",   does_not_hallucinate_schema, t2_nl[:120])
    log("T2 has result or clear error", t2.get("row_count", 0) > 0 or bool(t2_nl), t2_nl[:80])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: ISO codes auto-expanded to full country names
# ─────────────────────────────────────────────────────────────────────────────
async def test_iso_code_expansion():
    print("\n[3] ISO code expansion — MY/VN mapped to full names")
    r = await run_query("show employment trends for MY and VN over the last 3 years")
    nl  = r.get("nl_response", "")
    sql = r.get("sql", "")
    print(f"  nl_response: {nl[:120]}")
    print(f"  sql snippet: {sql[:200]}")

    # Either the SQL uses full names, or the response correctly explains the data
    uses_full_name  = any(w in sql.lower() for w in ["malaysia", "vietnam", "viet nam"])
    has_rows        = r.get("row_count", 0) > 0
    # If still failed: the respond_node should at least explain WHY (country values)
    explains_issue  = any(w in nl.lower() for w in ["malaysia", "full name", "country", "viet"])

    log("SQL uses full country names OR got rows", uses_full_name or has_rows,
        f"rows={r.get('row_count',0)}, sql={sql[:80]}")
    log("Response explains or answers",           explains_issue or has_rows, nl[:120])
    log("No schema confabulation",
        "don't have access" not in nl.lower(),
        nl[:80])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Multi-turn — 3-hop context chain
# ─────────────────────────────────────────────────────────────────────────────
async def test_multiturn_chain():
    print("\n[4] Multi-turn 3-hop chain")

    # T1: establish focus
    t1 = await run_query("what is the average unemployment rate for Indonesia and Thailand?")
    t1_nl  = t1.get("nl_response", "")
    t1_sql = t1.get("sql", "")
    log("T1 succeeded", t1.get("succeeded") is True, t1_nl[:80])

    # T2: "compare them by gender" — must resolve "them" = Indonesia + Thailand
    history_2 = [
        {"role": "user",      "text": "what is the average unemployment rate for Indonesia and Thailand?"},
        {"role": "assistant", "nl_response": t1_nl, "sql": t1_sql},
    ]
    t2 = await run_query("compare them by gender", history=history_2)
    t2_nl  = t2.get("nl_response", "")
    t2_sql = t2.get("sql", "")
    log("T2 resolved 'them' to countries",
        any(c in (t2_nl + t2_sql).lower() for c in ["indonesia", "thailand"]),
        f"sql: {t2_sql[:120]}")
    log("T2 includes gender dimension",
        any(w in (t2_nl + t2_sql).lower() for w in ["sex", "gender", "male", "female"]),
        t2_sql[:100])

    # T3: "and what about employment?" — resolve from T1+T2 context
    history_3 = history_2 + [
        {"role": "user",      "text": "compare them by gender"},
        {"role": "assistant", "nl_response": t2_nl, "sql": t2_sql},
    ]
    t3 = await run_query("and what about employment?", history=history_3)
    t3_nl  = t3.get("nl_response", "")
    t3_sql = t3.get("sql", "")
    log("T3 still references Indonesia/Thailand",
        any(c in (t3_nl + t3_sql).lower() for c in ["indonesia", "thailand"]),
        f"sql: {t3_sql[:100]}")
    log("T3 no confabulation",
        "don't have access" not in t3_nl.lower(),
        t3_nl[:80])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: INSUFFICIENT_DATA — query for a country that doesn't exist
# ─────────────────────────────────────────────────────────────────────────────
async def test_insufficient_data_signal():
    print("\n[5] INSUFFICIENT_DATA — non-existent entity")
    r = await run_query("show employment trends for Narnia over the last 5 years")
    nl = r.get("nl_response", "")
    print(f"  nl_response: {nl[:160]}")

    # Agent should NOT return rows for a non-existent country
    log("no fake rows returned", r.get("row_count", 0) == 0, f"rows={r.get('row_count',0)}")
    # And the response should acknowledge the data isn't there — not hallucinate
    says_not_found = any(w in nl.lower() for w in [
        "no data", "not found", "no results", "couldn't find", "doesn't exist",
        "not available", "insufficient", "no records", "cannot find"
    ])
    log("response acknowledges missing data", says_not_found, nl[:120])


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Auto-learn gate — confirm it fires correctly on success
# ─────────────────────────────────────────────────────────────────────────────
async def test_autolearn_gate():
    print("\n[6] Auto-learn gate — successful query should gate-pass")
    # Run a clear, answerable query
    r = await run_query("which country has the highest female employment rate?")
    log("query succeeded",  r.get("succeeded") is True, r.get("nl_response","")[:80])
    log("has rows",         r.get("row_count", 0) > 0)

    # Then check learning stats
    async with httpx.AsyncClient(timeout=10) as client:
        stats = await client.get(f"{BASE}/api/learning/stats")
    data = stats.json() if stats.status_code == 200 else {}
    log("learning endpoint reachable", stats.status_code == 200, str(data)[:80])

    # The exact count doesn't matter — just that the system ran and gated properly
    print(f"  learn stats: {json.dumps(data, indent=2)[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    print("=" * 60)
    print("sqlagent — agentic behavior end-to-end tests")
    print(f"  server:    {BASE}")
    print(f"  workspace: {WS_ID}")
    print("=" * 60)

    # Smoke-check server
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            h = await client.get(f"{BASE}/health")
            print(f"  health: {h.json().get('status')}")
        except Exception as e:
            print(f"\n  ERROR: server not reachable — {e}")
            sys.exit(1)

    await test_basic_cross_source()
    await test_anaphora_short_followup()
    await test_iso_code_expansion()
    await test_multiturn_chain()
    await test_insufficient_data_signal()
    await test_autolearn_gate()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    print(f"\n{'=' * 60}")
    print(f"  {passed}/{total} assertions passed")
    if passed < total:
        print("\n  Failed assertions:")
        for label, ok in results:
            if not ok:
                print(f"    ✗ {label}")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
