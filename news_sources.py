"""Abstracciones de proveedores de noticias para NB03.

Diseñado para desacoplar extracción de noticias, declarar calidad de cobertura
por proveedor y evitar asumir que cobertura histórica es equivalente entre fuentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import time
from typing import Iterable, Optional

import pandas as pd


STANDARD_COLUMNS = [
    "date",
    "headline",
    "source_name",
    "url",
    "provider",
    "coverage_quality",
]


@dataclass
class ProviderWindow:
    provider: str
    reliable_from: Optional[str]
    description: str


class BaseNewsSource:
    provider = "base"

    def fetch(self, search_terms: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError


class GDELTNewsSource(BaseNewsSource):
    provider = "gdelt"

    def __init__(
        self,
        cache_path: str,
        domains: Optional[list[str]] = None,
        reliable_from: str = "2017-01-01",
        sleep_s: float = 6.0,
        max_retries: int = 3,
        base_sleep: float = 10.0,
        block_pause: float = 8.0,
    ) -> None:
        self.cache_path = cache_path
        self.domains = domains or []
        self.reliable_from = pd.Timestamp(reliable_from)
        self.sleep_s = sleep_s
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.block_pause = block_pause

    @staticmethod
    def _quarter_blocks(start: str, end: str) -> list[tuple[str, str]]:
        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize()
        if t0 >= t1:
            return []
        out: list[tuple[str, str]] = []
        cur = t0
        while cur < t1:
            nxt = min(cur + pd.DateOffset(months=3), t1)
            out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
            cur = nxt
        return out

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        msg = repr(exc).lower()
        keys = ("ratelimit", "please limit requests", "too many requests")
        return any(k in msg for k in keys)

    def _search_with_retry(self, gd, term: str, t_start: str, t_end: str) -> pd.DataFrame:
        from gdeltdoc import Filters

        for attempt in range(self.max_retries + 1):
            try:
                kwargs = {"keyword": term, "start_date": t_start, "end_date": t_end}
                if self.domains:
                    kwargs["domain"] = self.domains
                return gd.article_search(Filters(**kwargs))
            except Exception as exc:
                if not self._is_rate_limit(exc) or attempt == self.max_retries:
                    print(f"  Error GDELT [{t_start}] '{term}': {repr(exc)}")
                    return pd.DataFrame()
                wait = self.base_sleep * (2**attempt)
                print(f"  Rate limit [{t_start}] '{term}' → retry {attempt + 1}/{self.max_retries} en {wait:.0f}s")
                time.sleep(wait)
        return pd.DataFrame()

    def fetch(self, search_terms: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        if os.path.exists(self.cache_path):
            df = pd.read_csv(self.cache_path)
            if not df.empty:
                return self._standardize(df)

        from gdeltdoc import GdeltDoc

        t0 = max(pd.Timestamp(start_date), self.reliable_from)
        t1 = min(pd.Timestamp(end_date), pd.Timestamp(datetime.now(timezone.utc).date()))
        blocks = self._quarter_blocks(str(t0.date()), str(t1.date()))
        if not blocks:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        gd = GdeltDoc()
        chunks: list[pd.DataFrame] = []
        for i, (b0, b1) in enumerate(blocks, 1):
            local = []
            for term in search_terms:
                part = self._search_with_retry(gd, term, b0, b1)
                if part is not None and not part.empty:
                    local.append(part)
                time.sleep(max(self.sleep_s, 5.0))
            if local:
                chunks.append(pd.concat(local, ignore_index=True))
            if self.block_pause > 0:
                time.sleep(self.block_pause)
            if i % 5 == 0 or i == len(blocks):
                print(f"  GDELT bloque {i}/{len(blocks)}")

        if not chunks:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        raw = pd.concat(chunks, ignore_index=True)
        raw.to_csv(self.cache_path, index=False)
        return self._standardize(raw)

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "seendate" in out.columns:
            out["date"] = pd.to_datetime(out["seendate"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
        elif "date" in out.columns:
            out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        else:
            out["date"] = pd.NaT

        out["headline"] = out.get("title", out.get("headline", "")).fillna("").astype(str)
        out["source_name"] = out.get("domain", out.get("source_name", "gdelt"))
        out["url"] = out.get("url", "")
        out["provider"] = self.provider
        out["coverage_quality"] = out["date"].apply(
            lambda d: "actual_news" if pd.notna(d) and d >= self.reliable_from else "low_confidence"
        )
        out = out[STANDARD_COLUMNS].dropna(subset=["date"]).drop_duplicates(subset=["date", "headline", "provider"])
        return out


class HistoricalNewsSource(BaseNewsSource):
    """Fuente histórica opcional basada en CSV local precurado."""

    provider = "historical_csv"

    def __init__(self, csv_path: str, quality: str = "actual_news") -> None:
        self.csv_path = csv_path
        self.quality = quality

    def fetch(self, search_terms: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        _ = search_terms
        if not self.csv_path or not os.path.exists(self.csv_path):
            return pd.DataFrame(columns=STANDARD_COLUMNS)
        df = pd.read_csv(self.csv_path)
        if "date" not in df.columns or "headline" not in df.columns:
            raise ValueError("HistoricalNewsSource requiere columnas 'date' y 'headline'.")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        out = out[(out["date"] >= pd.Timestamp(start_date)) & (out["date"] <= pd.Timestamp(end_date))]
        out["source_name"] = out.get("source_name", "historical_csv")
        out["url"] = out.get("url", "")
        out["provider"] = self.provider
        out["coverage_quality"] = self.quality
        out = out[STANDARD_COLUMNS].dropna(subset=["date", "headline"]).drop_duplicates(subset=["date", "headline", "provider"])
        return out


class NoNewsSource(BaseNewsSource):
    provider = "no_news"

    def fetch(self, search_terms: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        _ = (search_terms, start_date, end_date)
        return pd.DataFrame(columns=STANDARD_COLUMNS)


def build_news_sources(mode: str, cache_path: str, domains: list[str], historical_csv_path: str = "") -> list[BaseNewsSource]:
    mode = (mode or "auto").strip().lower()
    gdelt = GDELTNewsSource(cache_path=cache_path, domains=domains)
    historical = HistoricalNewsSource(csv_path=historical_csv_path) if historical_csv_path else None

    if mode == "gdelt_only":
        return [gdelt]
    if mode == "historical_plus_gdelt":
        return [s for s in [historical, gdelt] if s is not None]
    if mode == "auto":
        if historical is not None:
            return [historical, gdelt]
        return [gdelt]
    raise ValueError(f"news_source_mode no soportado: {mode}")


def fetch_news_from_sources(
    sources: Iterable[BaseNewsSource],
    search_terms: list[str],
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict]:
    parts = []
    provider_stats = {}
    for src in sources:
        got = src.fetch(search_terms=search_terms, start_date=start_date, end_date=end_date)
        got = got.copy()
        for col in STANDARD_COLUMNS:
            if col not in got.columns:
                got[col] = "" if col != "date" else pd.NaT
        got = got[STANDARD_COLUMNS]
        parts.append(got)
        provider_stats[src.provider] = {
            "rows": int(len(got)),
            "min_date": None if got.empty else str(got["date"].min().date()),
            "max_date": None if got.empty else str(got["date"].max().date()),
        }

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=STANDARD_COLUMNS)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["date", "provider"]).drop_duplicates(subset=["date", "headline", "provider"])

    meta = {
        "provider_stats": provider_stats,
        "requested_range": {"start": start_date, "end": end_date},
        "effective_range": {
            "start": None if df.empty else str(df["date"].min().date()),
            "end": None if df.empty else str(df["date"].max().date()),
        },
    }
    return df, meta
