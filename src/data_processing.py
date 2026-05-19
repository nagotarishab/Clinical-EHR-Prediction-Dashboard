import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _read_csv(path, **kwargs):
    kw = {"on_bad_lines": "skip", "low_memory": False}
    kw.update(kwargs)
    return pd.read_csv(path, **kw)


def _slug(s):
    return (
        str(s)
        .replace(" ", "_")
        .replace(")", "")
        .replace("(", "")
        .replace("[", "")
        .replace("]", "")
        .replace("/", "_")
        .replace("__", "_")
    )


class ClinicalDataProcessor:
    def __init__(
        self,
        data_dir,
        temporal_split_date="2018-01-01",
        target_condition="hypertension",
        test_size=0.2,
        random_state=42,
    ):
        self.data_dir = data_dir
        self.temporal_split_date = pd.to_datetime(temporal_split_date).tz_localize(None)
        self.target_condition = target_condition.lower()
        self.test_size = test_size
        self.random_state = random_state

        self.num_scaler = StandardScaler()
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.num_imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        self.cat_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)

        self.num_cols = []
        self.cat_cols = []

    def load_and_merge(self):
        """
        Load every CSV under synthea-mimic, integrate on PATIENT / ENCOUNTER / ORGANIZATION /
        PROVIDER / CLAIM linkage, aggregate longitudinal tables per encounter, and build labels.
        """
        logging.info("Loading all CSV files from %s", self.data_dir)

        patients = _read_csv(os.path.join(self.data_dir, "patients.csv"))
        encounters = _read_csv(os.path.join(self.data_dir, "encounters.csv"))
        conditions = _read_csv(os.path.join(self.data_dir, "conditions.csv"))
        observations = _read_csv(os.path.join(self.data_dir, "observations.csv"))
        medications = _read_csv(os.path.join(self.data_dir, "medications.csv"))
        procedures = _read_csv(os.path.join(self.data_dir, "procedures.csv"))
        allergies = _read_csv(os.path.join(self.data_dir, "allergies.csv"))
        devices = _read_csv(os.path.join(self.data_dir, "devices.csv"))
        careplans = _read_csv(os.path.join(self.data_dir, "careplans.csv"))
        immunizations = _read_csv(os.path.join(self.data_dir, "immunizations.csv"))
        imaging_studies = _read_csv(os.path.join(self.data_dir, "imaging_studies.csv"))
        supplies = _read_csv(os.path.join(self.data_dir, "supplies.csv"))
        organizations = _read_csv(os.path.join(self.data_dir, "organizations.csv"))
        providers = _read_csv(os.path.join(self.data_dir, "providers.csv"))
        payer_transitions = _read_csv(os.path.join(self.data_dir, "payer_transitions.csv"))
        claims = _read_csv(os.path.join(self.data_dir, "claims.csv"))
        claims_transactions = _read_csv(os.path.join(self.data_dir, "claims_transactions.csv"))

        self.merge_inspection = {
            "source_row_counts": {
                "patients.csv": len(patients),
                "encounters.csv": len(encounters),
                "conditions.csv": len(conditions),
                "observations.csv": len(observations),
                "medications.csv": len(medications),
                "procedures.csv": len(procedures),
                "allergies.csv": len(allergies),
                "devices.csv": len(devices),
                "careplans.csv": len(careplans),
                "immunizations.csv": len(immunizations),
                "imaging_studies.csv": len(imaging_studies),
                "supplies.csv": len(supplies),
                "organizations.csv": len(organizations),
                "providers.csv": len(providers),
                "payer_transitions.csv": len(payer_transitions),
                "claims.csv": len(claims),
                "claims_transactions.csv": len(claims_transactions),
            },
            "linkage_summary": (
                "Grain is **one row per encounter**. `PATIENT` joins demographics (race, gender, income), "
                "payer-transition counts, and encounter-level clinical tables. `ENCOUNTER` / encounter `Id` "
                "joins observations, medications, procedures, allergies, devices, careplans, immunizations, "
                "imaging, supplies, conditions (label), and claims via `APPOINTMENTID`."
            ),
        }

        logging.info("Processing base encounters...")
        encounters["START"] = pd.to_datetime(encounters["START"], errors="coerce").dt.tz_localize(None)
        encounters = encounters.dropna(subset=["START"]).copy()

        logging.info("Merging patient demographics...")
        pat = patients[["Id", "BIRTHDATE", "RACE", "ETHNICITY", "GENDER", "INCOME"]].copy()
        pat["BIRTHDATE"] = pd.to_datetime(pat["BIRTHDATE"], errors="coerce").dt.tz_localize(None)
        encounters = encounters.merge(pat, left_on="PATIENT", right_on="Id", how="left", suffixes=("", "_pat"))
        encounters["AGE_AT_ENCOUNTER"] = (encounters["START"] - encounters["BIRTHDATE"]).dt.days / 365.25
        encounters.drop(columns=["Id_pat", "BIRTHDATE"], errors="ignore", inplace=True)
        if "Id" in encounters.columns:
            encounters.rename(columns={"Id": "ENCOUNTER_ID"}, inplace=True)

        logging.info("Merging organizations and providers...")
        org = organizations.rename(columns={"Id": "ORG_LOOKUP_ID"})[
            ["ORG_LOOKUP_ID", "REVENUE", "UTILIZATION"]
        ].copy()
        for c in ["REVENUE", "UTILIZATION"]:
            org[c] = pd.to_numeric(org[c], errors="coerce")
        encounters = encounters.merge(
            org, left_on="ORGANIZATION", right_on="ORG_LOOKUP_ID", how="left"
        )

        prov = providers.rename(columns={"Id": "PROV_LOOKUP_ID"})[
            ["PROV_LOOKUP_ID", "SPECIALITY", "GENDER"]
        ].copy()
        prov.rename(columns={"GENDER": "PROVIDER_GENDER"}, inplace=True)
        encounters = encounters.merge(
            prov, left_on="PROVIDER", right_on="PROV_LOOKUP_ID", how="left"
        )

        logging.info("Payer transition counts per patient...")
        pt_counts = payer_transitions.groupby("PATIENT").size().rename("N_PAYER_TRANSITIONS").reset_index()
        encounters = encounters.merge(pt_counts, on="PATIENT", how="left")

        logging.info("Target variable from conditions...")
        desc_col = "DESCRIPTION" if "DESCRIPTION" in conditions.columns else conditions.columns[6]
        conditions["_desc"] = conditions[desc_col].astype(str).str.lower()
        target_cond_df = conditions[conditions["_desc"].str.contains(self.target_condition, na=False)]
        target_encounters = set(target_cond_df["ENCOUNTER"].dropna())
        encounters["TARGET"] = encounters["ENCOUNTER_ID"].apply(lambda x: 1 if x in target_encounters else 0)

        logging.info("Aggregating observations (mean + variance per encounter & lab type)...")
        obs_num = observations[observations["TYPE"] == "numeric"].copy()
        obs_num["VALUE"] = pd.to_numeric(obs_num["VALUE"], errors="coerce")
        obs_num = obs_num.dropna(subset=["VALUE", "ENCOUNTER"])
        key_obs = [
            "Body Height",
            "Body Weight",
            "Body mass index (BMI) [Ratio]",
            "Diastolic Blood Pressure",
            "Systolic Blood Pressure",
            "Heart rate",
            "Respiratory rate",
        ]
        obs_filtered = obs_num[obs_num["DESCRIPTION"].isin(key_obs)]
        g = obs_filtered.groupby(["ENCOUNTER", "DESCRIPTION"])["VALUE"].agg(["mean", "var"])
        obs_wide = g.unstack("DESCRIPTION")
        obs_wide.columns = [_slug(f"{a}__{b}") for a, b in obs_wide.columns]
        obs_wide = obs_wide.reset_index().rename(columns={"ENCOUNTER": "ENCOUNTER_OBS"})
        # Single observation per type → NaN var; treat as 0 variance (Welford would need n≥2).
        num_obs = obs_wide.select_dtypes(include=[np.number]).columns
        obs_wide[num_obs] = obs_wide[num_obs].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        logging.info("Encounter-level aggregates: medications, procedures, ...")
        med = medications.copy()
        for c in ["TOTALCOST", "BASE_COST", "PAYER_COVERAGE", "DISPENSES"]:
            if c in med.columns:
                med[c] = pd.to_numeric(med[c], errors="coerce")
        med = med.dropna(subset=["ENCOUNTER"])
        if len(med) == 0:
            med_agg = pd.DataFrame(columns=["ENCOUNTER_ID", "med_n"])
        else:
            med_agg = med.groupby("ENCOUNTER", as_index=False).agg(med_n=("ENCOUNTER", "count"))
            if "TOTALCOST" in med.columns:
                tsum = med.groupby("ENCOUNTER", as_index=False)["TOTALCOST"].sum().rename(
                    columns={"TOTALCOST": "med_totalcost"}
                )
                med_agg = med_agg.merge(tsum, on="ENCOUNTER", how="left")
            if "BASE_COST" in med.columns:
                bsum = med.groupby("ENCOUNTER", as_index=False)["BASE_COST"].sum().rename(
                    columns={"BASE_COST": "med_basecost"}
                )
                med_agg = med_agg.merge(bsum, on="ENCOUNTER", how="left")
            med_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        proc = procedures.copy()
        if "BASE_COST" in proc.columns:
            proc["BASE_COST"] = pd.to_numeric(proc["BASE_COST"], errors="coerce")
        proc = proc.dropna(subset=["ENCOUNTER"])
        if len(proc) == 0:
            proc_agg = pd.DataFrame(columns=["ENCOUNTER_ID", "proc_n"])
        else:
            proc_agg = proc.groupby("ENCOUNTER", as_index=False).agg(proc_n=("ENCOUNTER", "count"))
            if "BASE_COST" in proc.columns:
                ps = proc.groupby("ENCOUNTER", as_index=False)["BASE_COST"].sum().rename(
                    columns={"BASE_COST": "proc_basecost"}
                )
                proc_agg = proc_agg.merge(ps, on="ENCOUNTER", how="left")
            proc_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        all_agg = allergies.dropna(subset=["ENCOUNTER"]).groupby("ENCOUNTER").size().rename("allergy_n").reset_index()
        all_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        dev_agg = devices.dropna(subset=["ENCOUNTER"]).groupby("ENCOUNTER").size().rename("device_n").reset_index()
        dev_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        cp_agg = careplans.dropna(subset=["ENCOUNTER"]).groupby("ENCOUNTER").size().rename("careplan_n").reset_index()
        cp_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        imm = immunizations.copy()
        if "BASE_COST" in imm.columns:
            imm["BASE_COST"] = pd.to_numeric(imm["BASE_COST"], errors="coerce")
        imm = imm.dropna(subset=["ENCOUNTER"])
        if len(imm) == 0:
            imm_agg = pd.DataFrame(columns=["ENCOUNTER_ID", "imm_n"])
        else:
            imm_agg = imm.groupby("ENCOUNTER", as_index=False).agg(imm_n=("ENCOUNTER", "count"))
            if "BASE_COST" in imm.columns:
                isum = imm.groupby("ENCOUNTER", as_index=False)["BASE_COST"].sum().rename(
                    columns={"BASE_COST": "imm_basecost"}
                )
                imm_agg = imm_agg.merge(isum, on="ENCOUNTER", how="left")
            imm_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        img_agg = (
            imaging_studies.dropna(subset=["ENCOUNTER"])
            .groupby("ENCOUNTER")
            .size()
            .rename("imaging_n")
            .reset_index()
            .rename(columns={"ENCOUNTER": "ENCOUNTER_ID"})
        )

        sup = supplies.copy()
        if "QUANTITY" in sup.columns:
            sup["QUANTITY"] = pd.to_numeric(sup["QUANTITY"], errors="coerce")
        sup = sup.dropna(subset=["ENCOUNTER"])
        if len(sup) == 0:
            sup_agg = pd.DataFrame(columns=["ENCOUNTER_ID", "sup_n"])
        else:
            sup_agg = sup.groupby("ENCOUNTER", as_index=False).agg(sup_n=("ENCOUNTER", "count"))
            if "QUANTITY" in sup.columns:
                qsum = sup.groupby("ENCOUNTER", as_index=False)["QUANTITY"].sum().rename(
                    columns={"QUANTITY": "sup_qty"}
                )
                sup_agg = sup_agg.merge(qsum, on="ENCOUNTER", how="left")
            sup_agg.rename(columns={"ENCOUNTER": "ENCOUNTER_ID"}, inplace=True)

        logging.info("Claims and claim transactions linked by APPOINTMENTID -> encounter Id...")
        ct = claims_transactions.copy()
        if "AMOUNT" in ct.columns:
            ct["AMOUNT"] = pd.to_numeric(ct["AMOUNT"], errors="coerce")
            claim_amt = ct.groupby("CLAIMID")["AMOUNT"].sum().rename("claim_txn_amount_sum").reset_index()
            claim_amt.rename(columns={"CLAIMID": "Id"}, inplace=True)
            claims_m = claims.merge(claim_amt, on="Id", how="left")
        else:
            claims_m = claims.copy()
            claims_m["claim_txn_amount_sum"] = 0.0

        for c in ["OUTSTANDING1", "OUTSTANDING2", "OUTSTANDINGP"]:
            if c in claims_m.columns:
                claims_m[c] = pd.to_numeric(claims_m[c], errors="coerce")
        cm = claims_m.dropna(subset=["APPOINTMENTID"])
        if len(cm) == 0:
            agg_claim = pd.DataFrame(columns=["ENCOUNTER_ID"])
        else:
            agg_claim = cm.groupby("APPOINTMENTID", as_index=False).agg(claim_rows=("Id", "count"))
            if "OUTSTANDING1" in cm.columns:
                o = cm.groupby("APPOINTMENTID", as_index=False)["OUTSTANDING1"].max().rename(
                    columns={"OUTSTANDING1": "claim_outstanding1"}
                )
                agg_claim = agg_claim.merge(o, on="APPOINTMENTID", how="left")
            if "claim_txn_amount_sum" in cm.columns:
                t = cm.groupby("APPOINTMENTID", as_index=False)["claim_txn_amount_sum"].sum().rename(
                    columns={"claim_txn_amount_sum": "claim_txn_total"}
                )
                agg_claim = agg_claim.merge(t, on="APPOINTMENTID", how="left")
            agg_claim.rename(columns={"APPOINTMENTID": "ENCOUNTER_ID"}, inplace=True)

        logging.info("Merging feature tables to encounters...")
        df = encounters.merge(obs_wide, left_on="ENCOUNTER_ID", right_on="ENCOUNTER_OBS", how="left")
        df.drop(columns=["ENCOUNTER_OBS"], errors="ignore", inplace=True)

        for extra in (
            med_agg,
            proc_agg,
            all_agg,
            dev_agg,
            cp_agg,
            imm_agg,
            img_agg,
            sup_agg,
            agg_claim,
        ):
            df = df.merge(extra, on="ENCOUNTER_ID", how="left")

        self.cat_cols = [
            "RACE",
            "ETHNICITY",
            "GENDER",
            "ENCOUNTERCLASS",
            "SPECIALITY",
            "PROVIDER_GENDER",
        ]
        numeric_from_encounters = [
            "AGE_AT_ENCOUNTER",
            "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
            "PAYER_COVERAGE",
            "REVENUE",
            "UTILIZATION",
            "INCOME",
            "N_PAYER_TRANSITIONS",
        ]
        for c in numeric_from_encounters:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        non_feature = {
            "ENCOUNTER_ID",
            "START",
            "TARGET",
            "PATIENT",
            "ORGANIZATION",
            "PROVIDER",
            "PAYER",
            "ORG_LOOKUP_ID",
            "PROV_LOOKUP_ID",
            "REASONCODE",
            "DESCRIPTION",
            "CODE",
            "BASE_ENCOUNTER_COST",
            "TOTAL_CLAIM_COST",
            "PAYER_COVERAGE",
            "REVENUE",
            "UTILIZATION",
            "INCOME",
            "med_totalcost",
            "med_basecost",
            "proc_basecost",
            "imm_basecost",
            "claim_outstanding1",
            "claim_txn_total",
        }
        engineered = [
            c
            for c in df.columns
            if c not in non_feature and c not in self.cat_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        self.num_cols = [c for c in engineered if c != "TARGET"]

        n_patients_unified = int(df["PATIENT"].nunique()) if "PATIENT" in df.columns else 0
        self.merge_inspection["unified_encounters"] = int(len(df))
        self.merge_inspection["unique_patients"] = n_patients_unified

        select_cols = ["ENCOUNTER_ID", "START", "TARGET"] + self.cat_cols + self.num_cols
        select_cols = [c for c in select_cols if c in df.columns]
        df = df[select_cols].copy()

        miss = df.drop(columns=["START"], errors="ignore").isna().sum()
        self.merge_inspection["missing_cells_top"] = miss[miss > 0].sort_values(ascending=False).head(25)

        self.target_definition = (
            f"Binary **TARGET** = 1 iff this encounter’s `Id` appears in `conditions.csv` for a row whose "
            f"description contains **{self.target_condition!r}** (case-insensitive substring match)."
        )
        self.feature_justification = (
            "**Numeric:** age at encounter, encounter billing fields, organization utilization/revenue, "
            "patient income, payer-transition count, per-encounter counts/sums from meds/procedures/"
            "immunizations/supplies/claims, and **mean + variance** of key numeric vitals/labs aggregated "
            "within the encounter. **Categorical (one-hot after imputation):** race, ethnicity, gender, "
            "encounter class, provider speciality and gender. **Preprocessing:** mean imputation for numbers, "
            "most-frequent for categories, all fit on **Dataset 1 train** only; numerics are then "
            "**standardized** with the same scaler."
        )

        for c in self.num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.cat_cols:
            df[c] = df[c].astype(str).replace("nan", "Unknown")

        df[self.num_cols] = df[self.num_cols].replace([np.inf, -np.inf], np.nan)
        df["N_PAYER_TRANSITIONS"] = df["N_PAYER_TRANSITIONS"].fillna(0) if "N_PAYER_TRANSITIONS" in df.columns else None

        logging.info(
            "Final merged shape: %s, target prevalence: %.4f",
            df.shape,
            df["TARGET"].mean(),
        )
        return df

    def _stratify_labels(self, y):
        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2 or counts.min() < 2:
            return None
        return y

    def split_and_preprocess(self, df):
        """
        Temporal split -> Dataset 1 / Dataset 2, then train/test within each.
        Imputation, scaling, and one-hot encoding are fit on Dataset 1 train only.
        """
        logging.info("Temporal split at %s", self.temporal_split_date)
        df = df.dropna(subset=["START"]).sort_values(by="START")
        mask_hist = df["START"] < self.temporal_split_date
        df_hist = df[mask_hist].copy()
        df_curr = df[~mask_hist].copy()
        logging.info("Historical rows: %s, Current rows: %s", len(df_hist), len(df_curr))

        if len(df_hist) == 0:
            raise ValueError("Historical dataset is empty; choose a later temporal_split_date.")
        if len(df_curr) == 0:
            raise ValueError("Current dataset is empty; choose an earlier temporal_split_date.")

        strat_h = self._stratify_labels(df_hist["TARGET"].values)
        strat_c = self._stratify_labels(df_curr["TARGET"].values)

        df_h_train, df_h_test = train_test_split(
            df_hist,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat_h,
        )
        df_c_train, df_c_test = train_test_split(
            df_curr,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=strat_c,
        )

        logging.info("Fit preprocessors on Dataset 1 (historical) train only...")
        self.num_imputer.fit(df_h_train[self.num_cols])
        self.cat_imputer.fit(df_h_train[self.cat_cols])

        def _transform(frame):
            out = frame.copy()
            out[self.num_cols] = self.num_imputer.transform(out[self.num_cols])
            out[self.cat_cols] = self.cat_imputer.transform(out[self.cat_cols])
            return out

        df_h_train = _transform(df_h_train)
        df_h_test = _transform(df_h_test)
        df_c_train = _transform(df_c_train)
        df_c_test = _transform(df_c_test)

        # EDA / drift plots: numerics are still in natural units (years, costs, …), not z-scores.
        self.eda_num_h_test = df_h_test[self.num_cols].copy().reset_index(drop=True)
        self.eda_num_c_test = df_c_test[self.num_cols].copy().reset_index(drop=True)

        self.num_scaler.fit(df_h_train[self.num_cols])
        df_h_train[self.num_cols] = self.num_scaler.transform(df_h_train[self.num_cols])
        df_h_test[self.num_cols] = self.num_scaler.transform(df_h_test[self.num_cols])
        df_c_train[self.num_cols] = self.num_scaler.transform(df_c_train[self.num_cols])
        df_c_test[self.num_cols] = self.num_scaler.transform(df_c_test[self.num_cols])

        self.cat_encoder.fit(df_h_train[self.cat_cols])
        cat_h_tr = self.cat_encoder.transform(df_h_train[self.cat_cols])
        cat_h_te = self.cat_encoder.transform(df_h_test[self.cat_cols])
        cat_c_tr = self.cat_encoder.transform(df_c_train[self.cat_cols])
        cat_c_te = self.cat_encoder.transform(df_c_test[self.cat_cols])

        cat_names = list(self.cat_encoder.get_feature_names_out(self.cat_cols))
        feature_names = self.num_cols + cat_names

        def _stack(d, cat_part):
            X = np.hstack([d[self.num_cols].values.astype(np.float64), cat_part.astype(np.float64)])
            y = d["TARGET"].values.astype(np.int64)
            return X, y

        X_h_train, y_h_train = _stack(df_h_train, cat_h_tr)
        X_h_test, y_h_test = _stack(df_h_test, cat_h_te)
        X_c_train, y_c_train = _stack(df_c_train, cat_c_tr)
        X_c_test, y_c_test = _stack(df_c_test, cat_c_te)

        logging.info(
            "Shapes — D1 train %s, D1 test %s, D2 train %s, D2 test %s",
            X_h_train.shape,
            X_h_test.shape,
            X_c_train.shape,
            X_c_test.shape,
        )

        return (
            X_h_train,
            y_h_train,
            X_h_test,
            y_h_test,
            X_c_train,
            y_c_train,
            X_c_test,
            y_c_test,
            feature_names,
        )


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "synthea-mimic", "csv")
    processor = ClinicalDataProcessor(data_dir=os.path.abspath(data_dir))
    dfm = processor.load_and_merge()
    splits = processor.split_and_preprocess(dfm)
    print("Feature count:", len(splits[-1]))
