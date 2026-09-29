"""Ground-truth attack timeline for the CIC-IDS-2017 raw captures.

The MachineLearningCVE flow CSVs carry per-flow labels, but the raw PCAPs
do not. The dataset's authors publish the attack schedule instead (Sharafaldin
et al., "Toward Generating a New Intrusion Detection Dataset and Intrusion
Traffic Characterization", ICISSP 2018, and the CIC-IDS-2017 dataset page):
which attack ran when, from which attacker to which victim. This module turns
that schedule into a per-packet labeller -- the "attack timeline
annotations" the challenge asks supervised dynamics to be derived from.

Times are local to the testbed (UTC-3, Atlantic Daylight Time). A packet is
labelled with an attack when it falls inside the attack's interval AND its
endpoints are the documented attacker and victim. Inside the captured
network the external attacker (205.174.165.73) appears behind the firewall
as 172.16.0.1; both are accepted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

TESTBED_TZ = timezone(timedelta(hours=-3))
ATTACKERS = {"205.174.165.73", "172.16.0.1"}
WEB_SERVER = {"192.168.10.50", "205.174.165.68"}
HEARTBLEED_VICTIM = {"192.168.10.51", "205.174.165.66"}


def _t(day: str, hhmm: str) -> float:
    return datetime.strptime(f"{day} {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=TESTBED_TZ).timestamp()


# (label, start, end, attacker set, victim set)
TIMELINE = {
    "Monday": [],  # benign only
    "Tuesday": [
        ("FTP-Patator", _t("2017-07-04", "09:20"), _t("2017-07-04", "10:20"), ATTACKERS, WEB_SERVER),
        ("SSH-Patator", _t("2017-07-04", "14:00"), _t("2017-07-04", "15:00"), ATTACKERS, WEB_SERVER),
    ],
    "Wednesday": [
        ("DoS slowloris", _t("2017-07-05", "09:47"), _t("2017-07-05", "10:10"), ATTACKERS, WEB_SERVER),
        ("DoS Slowhttptest", _t("2017-07-05", "10:14"), _t("2017-07-05", "10:35"), ATTACKERS, WEB_SERVER),
        ("DoS Hulk", _t("2017-07-05", "10:43"), _t("2017-07-05", "11:00"), ATTACKERS, WEB_SERVER),
        ("DoS GoldenEye", _t("2017-07-05", "11:10"), _t("2017-07-05", "11:23"), ATTACKERS, WEB_SERVER),
        ("Heartbleed", _t("2017-07-05", "15:12"), _t("2017-07-05", "15:32"), ATTACKERS, HEARTBLEED_VICTIM),
    ],
    "Friday": [
        ("PortScan", _t("2017-07-07", "13:55"), _t("2017-07-07", "14:35"), ATTACKERS, WEB_SERVER),
        ("DDoS", _t("2017-07-07", "15:56"), _t("2017-07-07", "16:16"), ATTACKERS, WEB_SERVER),
    ],
}


def day_for(path: str) -> str | None:
    name = path.lower()
    for day in TIMELINE:
        if day.lower() in name:
            return day
    return None


def make_labeler(day: str):
    events = TIMELINE.get(day, [])
    lo = min((e[1] for e in events), default=None)
    hi = max((e[2] for e in events), default=None)

    def label(rec: dict) -> str | None:
        ts = rec["timestamp"]
        if lo is None or ts < lo or ts > hi:
            return None
        src, dst = rec["src_ip"], rec["dst_ip"]
        for name, start, end, attackers, victims in events:
            if start <= ts <= end and ((src in attackers and dst in victims) or (dst in attackers and src in victims)):
                return name
        return None

    return label
