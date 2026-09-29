"""Classification-tiered login and response authority."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.auth import tiered
from src.storage import blockchain_ledger


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def _key(username):
    return json.loads((tiered.AUTH_DIR / "keys" / f"{username}.kavachkey").read_text())


def _login(client, username):
    r = client.post("/auth/login", json={"username": username, "password": tiered.DEMO_PASSWORD}).json()
    while r["status"] == "step_required":
        if r["step"] == "totp":
            code = tiered.totp(tiered.get_user(username)["totp_secret"])
            r = client.post("/auth/verify", json={"challengeId": r["challengeId"], "code": code}).json()
        else:
            sig = tiered.key_response(_key(username)["secret"], r["nonce"])
            r = client.post("/auth/verify", json={"challengeId": r["challengeId"], "signature": sig}).json()
    return r


def _h(tok):
    return {"Authorization": f"Bearer {tok}"}


def test_primitives():
    h = tiered.hash_password("pw")
    assert tiered.check_password("pw", h) and not tiered.check_password("nope", h)
    # RFC 6238 SHA-1 test vector (secret "12345678901234567890", T=59 -> 94287082, 8 digits)
    sec = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    assert tiered.totp(sec, 59, digits=8) == "94287082"
    assert tiered.check_totp(sec, tiered.totp(sec))


def test_factor_count_scales_with_tier(client):
    r = client.post("/auth/login", json={"username": "l1.analyst", "password": "kavach2026"}).json()
    assert r["status"] == "ok" and r["user"]["tier"] == "low"
    r = client.post("/auth/login", json={"username": "l2.responder", "password": "kavach2026"}).json()
    assert r["status"] == "step_required" and r["step"] == "totp"
    r = client.post("/auth/login", json={"username": "officer.a", "password": "kavach2026"}).json()
    assert r["factors"] == ["password", "totp", "hardware_key"]
    code = tiered.totp(tiered.get_user("officer.a")["totp_secret"])
    r = client.post("/auth/verify", json={"challengeId": r["challengeId"], "code": code}).json()
    assert r["step"] == "hardware_key" and r["nonce"]
    bad = client.post("/auth/verify", json={"challengeId": r["challengeId"], "signature": "00" * 32})
    assert bad.status_code == 401
    assert _login(client, "officer.a")["user"]["factors"] == ["password", "totp", "hardware_key"]


def test_wrong_password_lockout_and_ledger(client):
    for _ in range(tiered.MAX_FAILURES):
        assert client.post("/auth/login", json={"username": "l1.analyst", "password": "x"}).status_code == 401
    assert client.post("/auth/login", json={"username": "l1.analyst", "password": "kavach2026"}).status_code == 423
    events = [b["data"]["event"] for b in blockchain_ledger.get_ledger().iter_blocks([blockchain_ledger.ACCESS])]
    assert events.count("LOGIN_FAILED") == tiered.MAX_FAILURES and "ACCOUNT_LOCKED" in events


def test_me_and_logout(client):
    tok = _login(client, "l2.responder")["token"]
    assert client.get("/auth/me", headers=_h(tok)).json()["tier"] == "moderate"
    client.post("/auth/logout", headers=_h(tok))
    assert client.get("/auth/me", headers=_h(tok)).status_code == 401


def test_response_rules_per_tier(client):
    act = {"action": "Block source IP", "target": "6.6.6.6", "command": "iptables -A INPUT -s 6.6.6.6 -j DROP"}
    assert client.post("/remediation/log", json=act).status_code == 401

    low = _login(client, "l1.analyst")["token"]
    r = client.post("/remediation/log", json=act, headers=_h(low)).json()
    assert r["status"] == "executed" and r["mode"] == "auto"

    mod = _login(client, "l2.responder")["token"]
    assert client.post("/remediation/log", json=act, headers=_h(mod)).status_code == 409
    r = client.post("/remediation/log", json={**act, "confirmed": True}, headers=_h(mod)).json()
    assert r["status"] == "executed" and r["mode"] == "analyst_confirmed"

    a = _login(client, "officer.a")["token"]
    b = _login(client, "officer.b")["token"]
    r = client.post("/remediation/log", json={**act, "confirmed": True}, headers=_h(a)).json()
    assert r["status"] == "pending_approval"
    rid = r["requestId"]
    assert client.get("/approvals/pending", headers=_h(b)).json()["requests"][0]["id"] == rid
    assert client.get("/approvals/pending", headers=_h(low)).json()["requests"] == []
    assert client.post(f"/approvals/{rid}", json={"decision": "approve"}, headers=_h(a)).status_code == 403  # own request
    assert client.post(f"/approvals/{rid}", json={"decision": "approve"}, headers=_h(mod)).status_code == 403  # not cleared
    r = client.post(f"/approvals/{rid}", json={"decision": "approve"}, headers=_h(b)).json()
    assert r["status"] == "executed" and r["block"]["data"]["approvers"] == ["officer.a@kavach.local", "officer.b@kavach.local"]
    assert blockchain_ledger.get_ledger().verify()["valid"]


def test_policy_and_demo_authenticator(client):
    p = client.get("/auth/policy").json()
    assert [t["tier"] for t in p["tiers"]] == ["low", "moderate", "high"]
    code = client.get("/auth/demo/authenticator/l2.responder").json()["code"]
    assert tiered.check_totp(tiered.get_user("l2.responder")["totp_secret"], code)
