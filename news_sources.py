"""Abstracciones de proveedores de noticias para NB03.

Diseñado para desacoplar extracción de noticias, declarar calidad de cobertura
por proveedor y evitar asumir que cobertura histórica es equivalente entre fuentes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
        """Genera bloques trimestrales con semántica inclusiva en `end`.

        Internamente usa intervalos semiabiertos [start, end+1día), lo que
        garantiza que start == end produzca un bloque válido.
        """
        t0 = pd.Timestamp(start).normalize()
        t1 = pd.Timestamp(end).normalize() + pd.Timedelta(days=1)
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

    def _cache_meta_path(self) -> str:
        return f"{self.cache_path}.meta.json"

    def _expected_cache_meta(self, search_terms: list[str]) -> dict:
        return {
            "provider": self.provider,
            "search_terms": sorted(search_terms),
            "domains": sorted(self.domains),
            "reliable_from": str(self.reliable_from.date()),
        }

    def _load_cache_meta(self) -> dict:
        path = self._cache_meta_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_cache_meta(self, meta: dict) -> None:
        with open(self._cache_meta_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _fetch_blocks(self, gd, blocks: list[tuple[str, str]], search_terms: list[str]) -> pd.DataFrame:
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
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)

    def fetch(self, search_terms: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        req_start = pd.Timestamp(start_date).normalize()
        req_end = pd.Timestamp(end_date).normalize()
        if req_start > req_end:
            raise ValueError(f"Rango inválido: {req_start.date()} > {req_end.date()}")

        upper_bound = pd.Timestamp(datetime.now(timezone.utc).date())
        query_start = max(req_start, self.reliable_from)
        query_end = min(req_end, upper_bound)
        if query_start > query_end:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        cache_df = pd.DataFrame(columns=STANDARD_COLUMNS)
        cache_meta_ok = False
        expected_meta = self._expected_cache_meta(search_terms)
        if os.path.exists(self.cache_path):
            try:
                loaded = pd.read_csv(self.cache_path)
                if not loaded.empty:
                    cache_df = self._standardize(loaded)
            except Exception:
                cache_df = pd.DataFrame(columns=STANDARD_COLUMNS)
            cache_meta_ok = self._load_cache_meta() == expected_meta

        if not cache_meta_ok:
            cache_df = pd.DataFrame(columns=STANDARD_COLUMNS)

        in_window = cache_df[
            (cache_df["date"] >= query_start) &
            (cache_df["date"] <= query_end)
        ] if not cache_df.empty else cache_df

        need_fetch = in_window.empty
        if not need_fetch:
            covered_min = pd.Timestamp(in_window["date"].min()).normalize()
            covered_max = pd.Timestamp(in_window["date"].max()).normalize()
            need_fetch = (covered_min > query_start) or (covered_max < query_end)

        fetched = pd.DataFrame()
        if need_fetch:
            from gdeltdoc import GdeltDoc
            gd = GdeltDoc()
            missing_blocks: list[tuple[str, str]] = []
            if in_window.empty:
                missing_blocks.extend(self._quarter_blocks(str(query_start.date()), str(query_end.date())))
            else:
                covered_min = pd.Timestamp(in_window["date"].min()).normalize()
                covered_max = pd.Timestamp(in_window["date"].max()).normalize()
                if query_start < covered_min:
                    missing_blocks.extend(self._quarter_blocks(str(query_start.date()), str((covered_min - pd.Timedelta(days=1)).date())))
                if covered_max < query_end:
                    missing_blocks.extend(self._quarter_blocks(str((covered_max + pd.Timedelta(days=1)).date()), str(query_end.date())))

            fetched_raw = self._fetch_blocks(gd=gd, blocks=missing_blocks, search_terms=search_terms) if missing_blocks else pd.DataFrame()
            fetched = self._standardize(fetched_raw) if not fetched_raw.empty else pd.DataFrame(columns=STANDARD_COLUMNS)

            combined = pd.concat([cache_df, fetched], ignore_index=True) if not cache_df.empty else fetched
            if combined.empty:
                combined = pd.DataFrame(columns=STANDARD_COLUMNS)
            else:
                combined = combined.sort_values(["date", "provider"]).drop_duplicates(subset=["date", "headline", "provider"])
            combined.to_csv(self.cache_path, index=False)
            self._write_cache_meta(expected_meta)
            cache_df = combined

        out = cache_df[
            (cache_df["date"] >= req_start) &
            (cache_df["date"] <= req_end)
        ] if not cache_df.empty else pd.DataFrame(columns=STANDARD_COLUMNS)
        return out.reset_index(drop=True)

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
