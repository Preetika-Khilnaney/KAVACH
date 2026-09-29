"""Classification-tiered authentication and response authority.

KAVACH runs one node per network classification tier. Login strength and the
authority to act on an alert both scale with the tier:

    tier      zone                                 login factors                    response rule
    low       Routine (office / admin)             password                         approved playbook auto-executes
    moderate  Sensitive (finance / HR / procure)   password + TOTP                  analyst must confirm each action
    high      Mission-critical (classified / OT)   password + TOTP + hardware key   two-person approval (or escalate)

Everything is offline and standard-library only:
  * passwords  -- PBKDF2-HMAC-SHA256, 200k iterations, per-user salt
  * TOTP       -- RFC 6238 (30 s step, 6 digits, +/-1 step drift), works with any authenticator app
  * hardware key -- challenge/response: the server issues a random nonce and the key
    proves possession of its secret with HMAC-SHA256(secret, nonce). In the prototype the
    "key" is a .kavachkey file (standing in for a FIDO2/PKI token); the secret never
    crosses the wire.
Every login success, failure and lockout is chained on the ledger as an ACCESS block.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import struct
import threading
import time
from pathlib import Path

AUTH_DIR = Path(os.environ.get("KAVACH_AUTH_DIR", "data/auth"))
DEMO_PASSWORD = "kavach2026"
MAX_FAILURES = 5
LOCKOUT_SECONDS = 300
CHALLENGE_TTL = 300

TIERS: dict[str, dict] = {
    "low": {"rank": 0, "label": "Routine", "zone": "Routine internal networks", "examples": "office · admin · staff email",
            "factors": ["password"], "response": "auto", "response_label": "Approved playbook actions auto-execute",
            "session_ttl": 8 * 3600},
    "moderate": {"rank": 1, "label": "Sensitive", "zone": "Sensitive networks", "examples": "finance · HR · procurement",
                 "factors": ["password", "totp"], "response": "confirm", "response_label": "Analyst confirms each action",
                 "session_ttl": 4 * 3600},
    "high": {"rank": 2, "label": "Mission-critical", "zone": "Mission-critical networks", "examples": "classified · defence · grid control",
             "factors": ["password", "totp", "hardware_key"], "response": "two_person",
             "response_label": "Two-person approval, or escalate to a senior officer", "session_ttl": 3600},
}

DEMO_USERS = [
    # username, display name, tier, role
    ("l1.analyst", "L1 Analyst", "low", "SOC analyst (L1)"),
    ("l2.responder", "L2 Responder", "moderate", "Incident responder (L2)"),
    ("officer.a", "Officer A", "high", "Cleared security officer"),
    ("officer.b", "Officer B", "high", "Cleared security officer"),
]

_lock = threading.RLock()
_sessions: dict[str, dict] = {}
_challenges: dict[str, dict] = {}
_failures: dict[str, dict] = {}
_approvals: dict[str, dict] = {}


class AuthError(Exception):
    def __init__(self, message: str, status: int = 401):
        super().__init__(message)
        self.status = status


# ── primitives ──────────────────────────────────────────────────────────────
def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"pbkdf2_sha256$200000${salt.hex()}${dk.hex()}"


def check_password(password: str, stored: str) -> bool:
    try:
        _, iters, salt, digest = stored.split("$")
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(dk.hex(), digest)
    except ValueError:
        return False


def totp(secret_b32: str, at: float | None = None, step: int = 30, digits: int = 6) -> str:
    key = base64.b32decode(secret_b32.upper() + "=" * (-len(secret_b32) % 8))
    counter = int((time.time() if at is None else at) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def check_totp(secret_b32: str, code: str, at: float | None = None) -> bool:
    now = time.time() if at is None else at
    code = (code or "").strip().replace(" ", "")
    return any(hmac.compare_digest(totp(secret_b32, now + d * 30), code) for d in (-1, 0, 1))


def key_response(key_secret_hex: str, nonce: str) -> str:
    return hmac.new(bytes.fromhex(key_secret_hex), nonce.encode(), hashlib.sha256).hexdigest()


# ── user store ──────────────────────────────────────────────────────────────
def _users_path() -> Path:
    return AUTH_DIR / "users.json"


def _ensure_store() -> dict:
    path = _users_path()
    if path.exists():
        return json.loads(path.read_text())
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    (AUTH_DIR / "keys").mkdir(exist_ok=True)
    users = {}
    for username, name, tier, role in DEMO_USERS:
        u = {"username": username, "name": name, "tier": tier, "role": role, "analystId": f"{username}@kavach.local",
             "password": hash_password(DEMO_PASSWORD)}
        if "totp" in TIERS[tier]["factors"]:
            u["totp_secret"] = base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")
        if "hardware_key" in TIERS[tier]["factors"]:
            key_id = secrets.token_hex(4)
            key_secret = secrets.token_hex(32)
            u["hardware_key"] = {"key_id": key_id, "secret": key_secret}
            (AUTH_DIR / "keys" / f"{username}.kavachkey").write_text(json.dumps(
                {"kavach_hardware_key": 1, "username": username, "key_id": key_id, "secret": key_secret}, indent=2))
        users[username] = u
    path.write_text(json.dumps(users, indent=2))
    return users


def get_user(username: str) -> dict | None:
    with _lock:
        return _ensure_store().get((username or "").strip().lower())


def public_user(u: dict) -> dict:
    t = TIERS[u["tier"]]
    return {"username": u["username"], "name": u["name"], "role": u["role"], "analystId": u["analystId"], "tier": u["tier"],
            "tierLabel": t["label"], "zone": t["zone"], "factors": t["factors"], "responseRule": t["response"],
            "responseLabel": t["response_label"]}


# ── ledger ──────────────────────────────────────────────────────────────────
def _chain(event: str, username: str, **extra) -> None:
    try:
        from src.storage.blockchain_ledger import get_ledger
        get_ledger().log_access(event=event, username=username, **extra)
    except Exception:  # never let an audit hiccup block login
        pass


# ── lockout ─────────────────────────────────────────────────────────────────
def _check_lock(username: str) -> None:
    f = _failures.get(username)
    if f and f.get("locked_until", 0) > time.time():
        raise AuthError(f"Account locked after {MAX_FAILURES} failed attempts. Try again in "
                        f"{int(f['locked_until'] - time.time())} s.", 423)


def _fail(username: str, stage: str, tier: str | None) -> None:
    f = _failures.setdefault(username, {"count": 0})
    f["count"] += 1
    _chain("LOGIN_FAILED", username, tier=tier, stage=stage, attempt=f["count"])
    if f["count"] >= MAX_FAILURES:
        f["locked_until"] = time.time() + LOCKOUT_SECONDS
        f["count"] = 0
        _chain("ACCOUNT_LOCKED", username, tier=tier, seconds=LOCKOUT_SECONDS)


# ── login flow ──────────────────────────────────────────────────────────────
def _next_step(u: dict, done: list[str]) -> dict:
    for factor in TIERS[u["tier"]]["factors"]:
        if factor not in done:
            cid = secrets.token_urlsafe(16)
            ch = {"username": u["username"], "done": done, "factor": factor, "expires": time.time() + CHALLENGE_TTL}
            if factor == "hardware_key":
                ch["nonce"] = secrets.token_hex(16)
            _challenges[cid] = ch
            out = {"status": "step_required", "step": factor, "challengeId": cid, "tier": u["tier"],
                   "tierLabel": TIERS[u["tier"]]["label"], "factors": TIERS[u["tier"]]["factors"], "completed": done}
            if factor == "hardware_key":
                out["nonce"] = ch["nonce"]
                out["keyId"] = u["hardware_key"]["key_id"]
            return out
    # all factors satisfied -> session
    t = TIERS[u["tier"]]
    token = secrets.token_urlsafe(32)
    _sessions[token] = {"username": u["username"], "tier": u["tier"], "expires": time.time() + t["session_ttl"],
                        "factors": done}
    _failures.pop(u["username"], None)
    _chain("LOGIN_SUCCESS", u["username"], tier=u["tier"], factors=done)
    return {"status": "ok", "token": token, "expiresIn": t["session_ttl"], "user": public_user(u)}


def start_login(username: str, password: str) -> dict:
    with _lock:
        uname = (username or "").strip().lower()
        _check_lock(uname)
        u = get_user(uname)
        if u is None or not check_password(password, u["password"]):
            _fail(uname, "password", u["tier"] if u else None)
            raise AuthError("Invalid username or password.")
        return _next_step(u, ["password"])


def continue_login(challenge_id: str, *, code: str | None = None, signature: str | None = None) -> dict:
    with _lock:
        ch = _challenges.pop(challenge_id, None)
        if ch is None or ch["expires"] < time.time():
            raise AuthError("Login step expired. Please start again.")
        _check_lock(ch["username"])
        u = get_user(ch["username"])
        if ch["factor"] == "totp":
            ok = bool(code) and check_totp(u["totp_secret"], code)
        elif ch["factor"] == "hardware_key":
            ok = bool(signature) and hmac.compare_digest(key_response(u["hardware_key"]["secret"], ch["nonce"]), signature.lower())
        else:
            ok = False
        if not ok:
            _fail(u["username"], ch["factor"], u["tier"])
            label = "authenticator code" if ch["factor"] == "totp" else "security key"
            raise AuthError(f"Invalid {label}.")
        return _next_step(u, ch["done"] + [ch["factor"]])


def session(token: str | None) -> dict | None:
    if not token:
        return None
    with _lock:
        s = _sessions.get(token)
        if s is None or s["expires"] < time.time():
            _sessions.pop(token, None)
            return None
        u = get_user(s["username"])
        return {**public_user(u), "factors": s["factors"], "expiresAt": s["expires"]}


def logout(token: str | None) -> None:
    with _lock:
        s = _sessions.pop(token or "", None)
        if s:
            _chain("LOGOUT", s["username"], tier=s["tier"])


def demo_authenticator_code(username: str) -> dict:
    """Prototype helper standing in for the analyst's authenticator app."""
    u = get_user(username)
    if u is None or "totp_secret" not in u:
        raise AuthError("No authenticator registered for this user.", 404)
    return {"username": u["username"], "code": totp(u["totp_secret"]), "validFor": 30 - int(time.time()) % 30}


def policy() -> dict:
    return {"tiers": [{"tier": k, **{kk: vv for kk, vv in v.items() if kk != "session_ttl"},
                       "sessionMinutes": v["session_ttl"] // 60} for k, v in TIERS.items()],
            "indicatorFlow": "upward-only (low → moderate → high); nothing flows from a higher tier to a lower one",
            "demoUsers": [{"username": u, "tier": t, "name": n} for u, n, t, _ in DEMO_USERS]}


# ── response authority ──────────────────────────────────────────────────────
def authorise_action(sess: dict, action: dict) -> dict:
    """Apply the tier's response rule to a remediation action.
    Returns {"status": "executed"|"pending_approval", ...}."""
    rule = TIERS[sess["tier"]]["response"]
    if rule == "auto":
        return {"status": "executed", "mode": "auto", "approvers": [sess["analystId"]]}
    if rule == "confirm":
        if not action.get("confirmed"):
            raise AuthError("Sensitive zone: the analyst must explicitly confirm this action.", 409)
        return {"status": "executed", "mode": "analyst_confirmed", "approvers": [sess["analystId"]]}
    with _lock:
        rid = secrets.token_hex(6)
        _approvals[rid] = {"id": rid, "requestedBy": sess["analystId"], "requestedByName": sess["name"], "tier": sess["tier"],
                           "action": action, "status": "pending", "createdAt": time.time()}
        return {"status": "pending_approval", "mode": "two_person", "requestId": rid,
                "detail": "Mission-critical zone: a second cleared officer must approve before this executes."}


def pending_approvals() -> list[dict]:
    with _lock:
        return sorted((a for a in _approvals.values() if a["status"] == "pending"), key=lambda a: a["createdAt"])


def decide(sess: dict, request_id: str, decision: str) -> dict:
    with _lock:
        req = _approvals.get(request_id)
        if req is None or req["status"] != "pending":
            raise AuthError("No pending request with that id.", 404)
        if TIERS[sess["tier"]]["rank"] < TIERS["high"]["rank"]:
            raise AuthError("Only cleared officers in the mission-critical tier can decide.", 403)
        if decision == "approve":
            if sess["analystId"] == req["requestedBy"]:
                raise AuthError("Two-person rule: the requester cannot approve their own action.", 403)
            req.update(status="approved", decidedBy=sess["analystId"], decidedAt=time.time())
        elif decision == "escalate":
            req.update(status="escalated", decidedBy=sess["analystId"], decidedAt=time.time())
        elif decision == "reject":
            req.update(status="rejected", decidedBy=sess["analystId"], decidedAt=time.time())
        else:
            raise AuthError("decision must be approve | reject | escalate", 422)
        return dict(req)


def reset_state() -> None:
    """Test helper."""
    with _lock:
        _sessions.clear(); _challenges.clear(); _failures.clear(); _approvals.clear()
