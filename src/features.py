"""Static context features and chronological behavior features (unlabeled)."""
from collections import OrderedDict
import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    a, b, c, d = [np.radians(pd.to_numeric(v, errors="coerce")) for v in (lat1,lon1,lat2,lon2)]
    x = np.sin((c-a)/2)**2 + np.cos(a)*np.cos(c)*np.sin((d-b)/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(x,0,1)))


class StaticFeatures:
    """Group statistics are fit *only* on training accounts."""
    base = ["log_amount", "distance_from_home_km", "hour_sin", "hour_cos", "day_sin", "day_cos"]
    cats = ["category", "merchant", "merchant_region"]
    def __init__(self):
        self.context_stats = {}
        self.median = {}
        self.iqr = {}
        self.cat_maps = {}
        self.numeric_cols = list(self.base)  # Exclude deterministic amount-relative duplicates to reduce reconstruction leakage

    def _base(self, df):
        out = df.copy()
        t = pd.to_datetime(out["time"])
        hour = t.dt.hour + t.dt.minute / 60
        dow = t.dt.dayofweek
        out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
        out["distance_from_home_km"] = haversine_km(out["home_lat"],out["home_lon"],out["merchant_lat"],out["merchant_lon"])
        out["distance_from_home_km"] = np.log1p(out["distance_from_home_km"].clip(lower=0))
        out["hour_sin"] = np.sin(2*np.pi*hour/24)
        out["hour_cos"] = np.cos(2*np.pi*hour/24)
        out["day_sin"] = np.sin(2*np.pi*dow/7)
        out["day_cos"] = np.cos(2*np.pi*dow/7)
        return out

    def fit(self, df):
        temp = self._base(df)
        for c in ("category", "merchant"):
            stats = temp.groupby(c)["log_amount"].agg(["median", "count"])
            q = temp.groupby(c)["log_amount"].quantile([.25,.75]).unstack()
            stats["iqr"] = (q[.75]-q[.25]).clip(lower=.1)
            self.context_stats[c] = stats
        for c in self.numeric_cols:
            values = pd.to_numeric(temp[c], errors="coerce")
            self.median[c] = float(values.median()) if values.notna().any() else 0.
            q1, q3 = values.quantile([.25,.75]) if values.notna().any() else (0., 1.)
            self.iqr[c] = max(float(q3-q1), .1)
        for c in self.cats:
            # 0 missing, 1 unseen, 2+ known. Fixed vocabulary from train only.
            self.cat_maps[c] = {v: i+2 for i,v in enumerate(sorted(temp[c].dropna().astype(str).unique()))}
        return self

    def _relative(self, df):
        out = df.copy()
        for c in ("category", "merchant"):
            stats = self.context_stats[c]
            grp = out[c].map(stats["median"])
            spread = out[c].map(stats["iqr"])
            out["amount_relative_to_" + c] = (out["log_amount"]-grp)/spread
        return out

    def transform(self, df):
        tmp = self._base(df)
        x = np.zeros((len(tmp),len(self.numeric_cols)), dtype=np.float32)
        obs = np.zeros_like(x)
        for j,c in enumerate(self.numeric_cols):
            s = pd.to_numeric(tmp[c],errors="coerce")
            ok = np.isfinite(s)
            obs[:,j] = ok.to_numpy(dtype=np.float32)
            x[:,j] = np.clip(((s-self.median[c])/self.iqr[c]).fillna(0),-12,12).to_numpy(np.float32)
        cat = np.zeros((len(tmp),len(self.cats)),dtype=np.int64)
        for j,c in enumerate(self.cats):
            cat[:,j] = tmp[c].map(self.cat_maps[c]).fillna(1).to_numpy(np.int64)
        return x, obs, cat

    @property
    def cardinalities(self):
        return [len(self.cat_maps[c])+2 for c in self.cats]


class RecentDistinct:
    def __init__(self,k):
        self.k=k; self.events=OrderedDict()
    def add(self,value,time):
        if pd.isna(value): return np.nan
        if value in self.events: del self.events[value]
        self.events[value]=time
        if len(self.events)>self.k: self.events.popitem(last=False)
        if len(self.events)<self.k: return np.nan
        return float((max(self.events.values())-min(self.events.values())).total_seconds()/60)


class TemporalFeatures:
    """Config-driven repeated-event and categorical-transition spans, one account at a time."""
    def __init__(self, categorical=("merchant", "merchant_region")):
        self.categorical=list(categorical)
        self.repeat=["minutes_since_previous", "span_last_3", "span_last_5"]
        self.groups={"repeat":list(self.repeat)}
        for c in self.categorical:
            self.groups[c]=[f"{c}_change_minutes",f"{c}_span_2",f"{c}_span_3"]
        self.columns=[v for arr in self.groups.values() for v in arr]

    def transform(self, frame):
        df=frame.sort_values(["account","time","tx_id"]).copy().reset_index(drop=True)
        df["minutes_since_previous"] = df.groupby("account")["time"].diff().dt.total_seconds()/60
        for n,k in ((3,2),(5,4)):
            df[f"span_last_{n}"] = (df["time"]-df.groupby("account")["time"].shift(k)).dt.total_seconds()/60
        for c in self.categorical:
            df[f"previous_{c}"] = df.groupby("account")[c].shift()
            changed = df[c].notna() & df[f"previous_{c}"].notna() & (df[c] != df[f"previous_{c}"])
            df[f"{c}_changed"] = changed
            df[f"{c}_change_minutes"] = df["minutes_since_previous"].where(changed)
            for k in (2,3):
                values=np.full(len(df),np.nan,dtype=float)
                for _, indexes in df.groupby("account",sort=False).groups.items():
                    state=RecentDistinct(k)
                    for index in indexes:
                        values[index]=state.add(df.at[index,c],df.at[index,"time"])
                df[f"{c}_span_{k}"] = values
        for c in self.columns:
            df.loc[df[c]<0,c] = np.nan
        return df


class TemporalScaler:
    def __init__(self, columns):
        self.columns=list(columns); self.stats={}
    def fit(self,df):
        for c in self.columns:
            vals=np.log1p(pd.to_numeric(df[c],errors="coerce").where(lambda x:x>=0))
            med=vals.median(); q1=vals.quantile(.25);q3=vals.quantile(.75)
            self.stats[c]=(float(med) if np.isfinite(med) else 0.,max(float(q3-q1),.1) if np.isfinite(q3-q1) else 1.)
        return self
    def transform(self,df):
        x=np.zeros((len(df),len(self.columns)),np.float32)
        observed=np.zeros_like(x)
        for j,c in enumerate(self.columns):
            s=pd.to_numeric(df[c],errors="coerce").where(lambda t:t>=0)
            observed[:,j] = np.isfinite(s).to_numpy(np.float32)
            t=np.log1p(s)
            med,iqr=self.stats[c]
            x[:,j] = np.clip(((t-med)/iqr).fillna(0),-10,10).to_numpy(np.float32)
        return x,observed
