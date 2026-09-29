"""Multi-dataset flow ingestion: CSE-CIC-IDS-2018, CIC-IDS-2017 full
(GeneratedLabelledFlows) and CTU-13 binetflow, auto-detected by header."""

import pandas as pd

from src.features import extract
from src.ingestion import flow_parser
from src.training.labels import infiltration_label, stage_label

CIC2018 = (
    "Dst Port,Protocol,Timestamp,Flow Duration,Tot Fwd Pkts,Tot Bwd Pkts,TotLen Fwd Pkts,TotLen Bwd Pkts,"
    "Flow IAT Mean,Flow IAT Std,Flow IAT Max,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,Label\n"
    "80,6,14/02/2018 08:31:01,1000000,3,2,120,80,250000,10,300000,0,1,0,0,1,Benign\n"
    "22,6,14/02/2018 08:31:03,2000000,10,10,900,1800,200000,5,220000,1,1,0,1,1,SSH-Bruteforce\n"
    "53,17,14/02/2018 08:31:09,500,1,1,40,120,500,0,500,0,0,0,0,0,Benign\n"
)
CIC2017_FULL = (
    "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Protocol, Timestamp, Flow Duration,"
    " Total Fwd Packets, Total Backward Packets,Total Length of Fwd Packets, Total Length of Bwd Packets, Label\n"
    "a,172.16.0.1,4444,192.168.10.50,80,6,05/07/2017 09:47:10,1000,3,2,100,300,DoS slowloris\n"
    "b,192.168.10.5,5555,8.8.8.8,53,17,05/07/2017 09:47:12,1000,1,1,40,80,BENIGN\n"
)
CTU13 = (
    "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,Label\n"
    "2011/08/10 09:46:53.047277,3.1,tcp,147.32.84.165,1027,   ->,60.190.222.139,80,FSPA_FSPA,0,0,10,1500,600,flow=From-Botnet-V42-TCP-CC6-Plain-HTTP-Encrypted-Data\n"
    "2011/08/10 09:46:54.000000,0.01,udp,147.32.84.10,0x0303,  <->,147.32.80.9,53,CON,0,0,2,200,80,flow=From-Normal-V42-UDP-DNS\n"
    "2011/08/10 09:46:59.000000,0.2,tcp,147.32.84.20,4000,   ->,1.2.3.4,443,S_RA,0,0,3,180,120,flow=Background-TCP-Attempt\n"
)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_detect_formats(tmp_path):
    assert flow_parser.detect_format(_write(tmp_path, "a.csv", CIC2018)) == "cicids2018"
    assert flow_parser.detect_format(_write(tmp_path, "b.csv", CIC2017_FULL)) == "cicids2017_full"
    assert flow_parser.detect_format(_write(tmp_path, "c.binetflow", CTU13)) == "ctu13"


def test_cicids2018_uses_real_timestamps_protocol_and_flags(tmp_path):
    df = flow_parser.load_flow_csv(_write(tmp_path, "Wednesday-14-02-2018.csv", CIC2018))
    assert df["timestamp"].iloc[0] == pd.Timestamp("2018-02-14 08:31:01")
    assert df["timestamp"].iloc[2] == pd.Timestamp("2018-02-14 08:31:09")
    assert df["protocol"].tolist() == ["TCP", "TCP", "UDP"]
    assert df["bytes"].tolist() == [200, 2700, 160]
    assert df["tcp_flags"].iloc[1] == 0x01 | 0x02 | 0x08 | 0x10
    assert stage_label({"flow_labels_present": ["SSH-Bruteforce"]}) == 2  # Initial Access
    win = extract.build_feature_windows(flow_df=df, window_seconds=5, normalize=False)
    assert {"flow_bwd_fwd_byte_ratio", "flow_udp_fraction", "flow_tcp_fraction"} <= set(win.columns)
    assert len(win) == 2  # 08:31:00-05 and 08:31:05-10 (real time windows)


def test_cicids2017_full_keeps_ips(tmp_path):
    df = flow_parser.load_flow_csv(_write(tmp_path, "Wednesday.csv", CIC2017_FULL))
    assert df["src_ip"].tolist() == ["172.16.0.1", "192.168.10.5"]
    assert df["dst_ip"].tolist() == ["192.168.10.50", "8.8.8.8"]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2017-07-05 09:47:10")


def test_ctu13_binetflow(tmp_path):
    df = flow_parser.load_flow_csv(_write(tmp_path, "capture20110810.binetflow", CTU13))
    assert df["label"].tolist() == ["Botnet", "BENIGN", "BENIGN"]
    assert df["src_port"].iloc[1] == 0x0303
    assert df["tcp_flags"].iloc[0] == 0x01 | 0x02 | 0x08 | 0x10
    assert df["tcp_flags"].iloc[1] == 0  # UDP
    assert df["bwd_bytes"].iloc[0] == 900
    win = extract.build_feature_windows(flow_df=df, window_seconds=5, normalize=False)
    assert infiltration_label(win.iloc[0].to_dict()) == 1.0
    assert stage_label({"flow_labels_present": ["BENIGN", "Botnet"]}) == 4  # Command and Control
