"""Parse the ECCC PFAS wastewater/biosolids CSVs into tidy long tables.

Raw files are never modified. Output: data_clean/*.parquet (+ csv summaries).
Rules
- Data block = header row 'SAMPLE_NAME,...' until the first fully empty / dictionary row.
- Keep concentration units only (ng/L, ng/g); recovery rows are dropped from the
  environmental table but counted.
- Non-detect (LAB_FLAG contains U or value 'Non-Detect') -> left-censored at DL.
- Environmental matrices only (no blanks / Not Applicable).
- Duplicates (SAMPLE_NAME contains 'duplicate') are flagged, not averaged here.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "raw"
OUT = Path(__file__).resolve().parents[1] / "data_clean"
OUT.mkdir(exist_ok=True)

COLS = ["SAMPLE_NAME", "WWTP_CODE", "LOCATION", "NUMBER", "AXYS_ID", "COMPOUND",
        "CAS_NUMBER", "LAB_FLAG", "CONCENTRATION_FOUND", "DETECTION_LIMIT", "UNIT",
        "COLLECTION_DATE", "WORKGROUP"]


def read_block(fname):
    txt = (RAW / fname).read_bytes().decode("cp1252")
    lines = txt.splitlines()
    h = next(i for i, l in enumerate(lines) if l.startswith("SAMPLE_NAME,WWTP_CODE"))
    body = []
    for l in lines[h + 1:]:
        if not l.strip(",").strip():
            break
        body.append(l)
    df = pd.read_csv(io.StringIO("\n".join([lines[h]] + body)), dtype=str)
    return df[COLS]


def tidy(df, medium):
    d = df.copy()
    d["medium"] = medium
    d["is_recovery"] = d["UNIT"].str.contains("Recovery", case=False, na=False)
    d = d[~d["is_recovery"]].copy()
    d["date"] = pd.to_datetime(d["COLLECTION_DATE"], format="%Y-%m-%d", errors="coerce")
    flag = d["LAB_FLAG"].fillna("")
    raw = d["CONCENTRATION_FOUND"].fillna("")
    d["censored"] = raw.str.contains("Non-Detect", case=False) | flag.str.contains(r"(^|[^A-Za-z])U", regex=True)
    d["value"] = pd.to_numeric(raw.str.replace(",", ""), errors="coerce")
    d["dl"] = pd.to_numeric(d["DETECTION_LIMIT"].str.replace(",", ""), errors="coerce")
    d["is_dup"] = d["SAMPLE_NAME"].str.contains("duplicate", case=False, na=False)
    d["flag_H"] = flag.str.contains("H")
    d["flag_K"] = flag.str.contains("K")
    d["flag_B"] = flag.str.contains("B")
    d["flag_NQ"] = flag.str.contains("NQ")
    return d


def main():
    ww = read_block("PFASWastewater_EN_FR.csv")
    bs = read_block("PFASBiosolids_EN_FR.csv")
    print("raw rows", len(ww), len(bs))
    t = pd.concat([tidy(ww, "water"), tidy(bs, "solid")], ignore_index=True)
    print(t["LOCATION"].value_counts())
    print(t["UNIT"].value_counts())
    env_loc = ["Influent", "Effluent", "Primary Effluent", "Leachate", "Raw Leachate",
               "Treated Leachate", "Biosolids", "Primary Sludge", "Waste Activated Sludge",
               "Waste Secondary Sludge", "Sludge"]
    print("unmatched locations:", sorted(set(t["LOCATION"]) - set(env_loc)))
    env = t[t["LOCATION"].isin(env_loc) & t["date"].notna()].copy()
    # uncensored rows lacking a numeric value (NQ etc.)
    env["usable"] = env["censored"] & env["dl"].notna() | (~env["censored"] & env["value"].notna())
    print("usable", env["usable"].mean(), "censored", env["censored"].mean())
    env.to_parquet(OUT / "env_long.parquet")
    t.to_parquet(OUT / "all_conc_long.parquet")

    # plant descriptors
    raw = (RAW / "CMP_Master_WWTP_Code_List_EN_FR.csv").read_bytes().decode("cp1252")
    pl = pd.read_csv(io.StringIO(raw), skiprows=1, dtype=str).iloc[:, :7]
    pl.columns = ["code", "liquid", "solids", "receiving", "flow_avg", "pct_res", "pct_ici"]
    pl = pl[pl["code"].notna()]
    pl.to_csv(OUT / "plants.csv", index=False)

    raw = (RAW / "CMP_Master_Temp_Flows_EN_FR.csv").read_bytes().decode("cp1252")
    tf = pd.read_csv(io.StringIO(raw), skiprows=1, dtype=str).iloc[:, :7]
    tf.columns = ["code", "date", "flow", "leach_flow", "T_inf", "T_eff", "notes"]
    tf = tf[tf["code"].notna()]
    tf["date"] = pd.to_datetime(tf["date"], format="%m/%d/%Y", errors="coerce")
    for c in ["flow", "leach_flow", "T_inf", "T_eff"]:
        tf[c] = pd.to_numeric(tf[c].str.replace(",", ""), errors="coerce")
    tf.to_csv(OUT / "temp_flow.csv", index=False)
    print(tf.describe())


if __name__ == "__main__":
    main()
