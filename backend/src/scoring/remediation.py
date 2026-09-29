"""Stage-specific remediation playbooks (roadmap Task 6.2).

Each kill-chain stage maps to prioritised actions: what to do, why, a
suggested command (templated with the real IPs from the alert when they are
known), and the MITRE ATT&CK mitigation it corresponds to. Commands are
suggestions for a human operator -- nothing here executes them.
"""

from __future__ import annotations

import ipaddress

PRIORITY_ORDER = ["immediate", "short_term", "ongoing", "investigate"]
PRIORITY_LABELS = {"immediate": "Immediate", "short_term": "Short-term", "ongoing": "Ongoing", "investigate": "Investigate"}

_PLAYBOOK: dict[str, list[dict]] = {
    "Reconnaissance": [
        {"priority": "immediate", "action": "Block the scanning source at the perimeter firewall",
         "rationale": "Stops the adversary from completing its map of exposed services before it chooses an entry point.",
         "command": "iptables -I INPUT -s {src_ip} -j DROP", "mitigation": "M1037", "mitigation_name": "Filter Network Traffic"},
        {"priority": "short_term", "action": "Review firewall rules for services exposed to the scanned subnet",
         "rationale": "Every port that answered the scan is a candidate for initial access; close anything not required.",
         "command": "nmap -sS -p- --open {subnet}", "mitigation": "M1042", "mitigation_name": "Disable or Remove Feature or Program"},
        {"priority": "ongoing", "action": "Enable enhanced connection logging for the targeted hosts",
         "rationale": "Reconnaissance is usually followed by exploitation within minutes-to-hours; full logs make the next stage visible.",
         "command": "zeek -i eth0 -C 'Site::local_nets += {{ {subnet} }}' local", "mitigation": "M1031", "mitigation_name": "Network Intrusion Prevention"},
        {"priority": "investigate", "action": "Correlate the source with threat-intel feeds and other KAVACH nodes",
         "rationale": "A scanner seen at several sites is far more likely to be a coordinated campaign than a misconfigured tool.",
         "command": None, "mitigation": "M1019", "mitigation_name": "Threat Intelligence Program"},
    ],
    "Initial Access": [
        {"priority": "immediate", "action": "Isolate the targeted host from the network",
         "rationale": "Contain a possibly compromised host before the adversary establishes persistence or moves laterally.",
         "command": "iptables -I FORWARD -s {dst_ip} -j DROP && iptables -I FORWARD -d {dst_ip} -j DROP", "mitigation": "M1030", "mitigation_name": "Network Segmentation"},
        {"priority": "immediate", "action": "Capture a forensic PCAP of the host's traffic",
         "rationale": "Preserves evidence of the exploit attempt for attribution and for verifying the on-chain hash later.",
         "command": "tcpdump -i any -w /var/forensics/{dst_ip}_$(date +%s).pcap host {dst_ip}", "mitigation": "M1057", "mitigation_name": "Data Loss Prevention"},
        {"priority": "short_term", "action": "Patch or disable the exploited service and force credential resets",
         "rationale": "Brute-force and web exploits target known weaknesses; removing them closes the door for re-entry.",
         "command": None, "mitigation": "M1051", "mitigation_name": "Update Software"},
        {"priority": "investigate", "action": "Review authentication logs for successful logins after failed attempts",
         "rationale": "A success after a burst of failures is the signature of a working brute-force.",
         "command": "grep -E 'Accepted|Failed' /var/log/auth.log | grep {src_ip}", "mitigation": "M1036", "mitigation_name": "Account Use Policies"},
    ],
    "Lateral Movement": [
        {"priority": "immediate", "action": "Segment the affected subnet and block east-west traffic from the pivot host",
         "rationale": "Lateral movement turns one compromise into many; segmentation caps the blast radius.",
         "command": "iptables -I FORWARD -s {src_ip} -d {subnet} -j DROP", "mitigation": "M1030", "mitigation_name": "Network Segmentation"},
        {"priority": "immediate", "action": "Rotate credentials used on the affected hosts",
         "rationale": "Adversaries move laterally with harvested credentials; rotation invalidates what they collected.",
         "command": None, "mitigation": "M1027", "mitigation_name": "Password Policies"},
        {"priority": "short_term", "action": "Deploy honeypot services in the segment",
         "rationale": "Decoys surface further lateral probing with near-zero false positives.",
         "command": None, "mitigation": "M1054", "mitigation_name": "Software Configuration"},
        {"priority": "investigate", "action": "Enumerate every host the pivot contacted in the last 24h",
         "rationale": "Each one is potentially compromised and needs triage.",
         "command": "zeek-cut id.orig_h id.resp_h < conn.log | grep {src_ip} | sort -u", "mitigation": "M1047", "mitigation_name": "Audit"},
    ],
    "Command and Control": [
        {"priority": "immediate", "action": "Block the C2 endpoint at egress",
         "rationale": "Cutting the command channel stops the adversary from issuing new instructions to implants.",
         "command": "iptables -I OUTPUT -d {dst_ip} -j DROP && iptables -I FORWARD -d {dst_ip} -j DROP", "mitigation": "M1037", "mitigation_name": "Filter Network Traffic"},
        {"priority": "immediate", "action": "Identify all internal hosts beaconing to the same endpoint",
         "rationale": "Beaconing is periodic -- every host with the same rhythm to the same destination is likely implanted.",
         "command": "zeek-cut id.orig_h id.resp_h < conn.log | awk '$2==\"{dst_ip}\"' | sort -u", "mitigation": "M1031", "mitigation_name": "Network Intrusion Prevention"},
        {"priority": "short_term", "action": "Inspect egress for DNS / HTTPS tunnelling from affected hosts",
         "rationale": "C2 often fails over to covert channels once the primary endpoint is blocked.",
         "command": None, "mitigation": "M1020", "mitigation_name": "SSL/TLS Inspection"},
        {"priority": "investigate", "action": "Image the affected hosts' memory for implant analysis",
         "rationale": "Implant capabilities determine what data may already be at risk.",
         "command": None, "mitigation": "M1047", "mitigation_name": "Audit"},
    ],
    "Exfiltration": [
        {"priority": "immediate", "action": "Block all outbound traffic from the exfiltrating host",
         "rationale": "Every additional second of transfer is more classified data lost.",
         "command": "iptables -I OUTPUT -s {src_ip} -j DROP && iptables -I FORWARD -s {src_ip} -j DROP", "mitigation": "M1037", "mitigation_name": "Filter Network Traffic"},
        {"priority": "immediate", "action": "Preserve traffic logs and flow records for the transfer window",
         "rationale": "Establishes exactly what left and when -- the ledger hash proves the logs were not altered afterwards.",
         "command": "tcpdump -i any -w /var/forensics/exfil_{src_ip}_$(date +%s).pcap host {src_ip}", "mitigation": "M1057", "mitigation_name": "Data Loss Prevention"},
        {"priority": "short_term", "action": "Assess the sensitivity of data reachable from the host",
         "rationale": "Drives the damage assessment and mandatory reporting chain.",
         "command": None, "mitigation": "M1041", "mitigation_name": "Encrypt Sensitive Information"},
        {"priority": "investigate", "action": "Trace the kill chain back to the initial access vector",
         "rationale": "Exfiltration is the end of the chain; the entry point must be closed to prevent a repeat.",
         "command": None, "mitigation": "M1047", "mitigation_name": "Audit"},
    ],
}


def _subnet(ip: str | None) -> str:
    try:
        return str(ipaddress.ip_network(f"{ip}/24", strict=False))
    except (ValueError, TypeError):
        return "<subnet>"


def remediation_for_stage(stage: str, context: dict | None = None) -> list[dict]:
    """Prioritised actions for one stage. context may carry src_ip, dst_ip,
    host_ips; commands are filled with them (placeholders otherwise)."""
    context = context or {}
    hosts = list(context.get("host_ips") or [])
    src = context.get("src_ip") or (hosts[0] if hosts else None)
    dst = context.get("dst_ip") or (hosts[1] if len(hosts) > 1 else src)
    fill = {"src_ip": src or "<src_ip>", "dst_ip": dst or "<dst_ip>", "subnet": _subnet(dst or src)}
    actions = []
    for a in _PLAYBOOK.get(stage, []):
        item = dict(a)
        item["priority_label"] = PRIORITY_LABELS[item["priority"]]
        item["stage"] = stage
        if item.get("command"):
            item["command"] = item["command"].format(**fill)
        item["uses_real_ips"] = bool(src)
        actions.append(item)
    return sorted(actions, key=lambda x: PRIORITY_ORDER.index(x["priority"]))


def playbook_stages() -> list[str]:
    return list(_PLAYBOOK)
