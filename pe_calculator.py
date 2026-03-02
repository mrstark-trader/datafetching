"""Trailing TTM PE Calculator with 756-Day PE Bands
============================================================
1.  Uses **yfinance** ``earnings_history`` + ``quarterly_income_stmt``
    for **consolidated** quarterly EPS, falling back to
    ``get_earnings_dates`` Reported EPS for older quarters.
2.  Computes **TTM EPS** = sum of most recent 4 quarters' EPS.
3.  Fetches **daily closing prices** from **Fyers** API.
4.  **Trailing PE** = Close Price on result date / TTM EPS.
5.  **PE High / Low / Median** computed from daily PE values
    (daily close / applicable TTM EPS) over the preceding 756 days.

Single-symbol mode
    python pe_calculator.py --symbol RELIANCE
    python pe_calculator.py --symbol SBIN --quarters 8

Batch mode  (reads one symbol per line from symbols.txt)
    python pe_calculator.py --batch
    python pe_calculator.py --batch --file my_symbols.txt
"""

import sys
import os
import json
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd

# ---------------------------------------------------------------------------
# Project-path setup — reuse existing Fyers integration
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent        # trading_system/
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_fyers_old_path = _project_root / "Fyers_OLD"
if str(_fyers_old_path) not in sys.path:
    sys.path.insert(0, str(_fyers_old_path))

from Fyers_Service.FyersService import FyersService
from logger.LoggerInitializer import LoggerInitializer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYMBOL_MAP: Dict[str, Dict[str, str]] = {
    "SBIN":       {"yf": "SBIN.NS",       "fyers": "NSE:SBIN-EQ"},
    "RELIANCE":   {"yf": "RELIANCE.NS",   "fyers": "NSE:RELIANCE-EQ"},
    "TCS":        {"yf": "TCS.NS",        "fyers": "NSE:TCS-EQ"},
    "INFY":       {"yf": "INFY.NS",       "fyers": "NSE:INFY-EQ"},
    "HDFCBANK":   {"yf": "HDFCBANK.NS",   "fyers": "NSE:HDFCBANK-EQ"},
    "ICICIBANK":  {"yf": "ICICIBANK.NS",  "fyers": "NSE:ICICIBANK-EQ"},
    "ITC":        {"yf": "ITC.NS",        "fyers": "NSE:ITC-EQ"},
    "KOTAKBANK":  {"yf": "KOTAKBANK.NS",  "fyers": "NSE:KOTAKBANK-EQ"},
    "LT":         {"yf": "LT.NS",         "fyers": "NSE:LT-EQ"},
    "AXISBANK":   {"yf": "AXISBANK.NS",   "fyers": "NSE:AXISBANK-EQ"},
    "HDFC":       {"yf": "HDFC.NS",       "fyers": "NSE:HDFC-EQ"},
    "MARUTI":     {"yf": "MARUTI.NS",     "fyers": "NSE:MARUTI-EQ"},
    "WIPRO":      {"yf": "WIPRO.NS",      "fyers": "NSE:WIPRO-EQ"},
    "TATAMOTORS": {"yf": "TATAMOTORS.NS", "fyers": "NSE:TATAMOTORS-EQ"},
    "TATASTEEL":  {"yf": "TATASTEEL.NS",  "fyers": "NSE:TATASTEEL-EQ"},
    "BAJFINANCE": {"yf": "BAJFINANCE.NS", "fyers": "NSE:BAJFINANCE-EQ"},
    "SUNPHARMA":  {"yf": "SUNPHARMA.NS",  "fyers": "NSE:SUNPHARMA-EQ"},
    "HCLTECH":    {"yf": "HCLTECH.NS",    "fyers": "NSE:HCLTECH-EQ"},
    "ADANIENT":   {"yf": "ADANIENT.NS",   "fyers": "NSE:ADANIENT-EQ"},
    "BHARTIARTL": {"yf": "BHARTIARTL.NS", "fyers": "NSE:BHARTIARTL-EQ"},
}

DEFAULT_SYMBOLS_FILE = Path(__file__).parent / "symbols.txt"
OUTPUT_DIR = Path(__file__).parent / "output"

# How many quarters of earnings to fetch from yfinance
_EARNINGS_LIMIT = 50
# Days of lookback for PE High/Low/Median bands
_PE_BAND_DAYS = 756


# ---------------------------------------------------------------------------
# Earnings Client  (consolidated quarterly EPS from Yahoo Finance)
# ---------------------------------------------------------------------------
class EarningsClient:
    """
    Fetches quarterly **consolidated** EPS from Yahoo Finance.

    Uses a hybrid approach (best source wins):
    1. ``earnings_history``       — consolidated ``epsActual``  (~4 quarters)
    2. ``quarterly_income_stmt``  — Net Income / sharesOutstanding (~6 quarters)
    3. ``get_earnings_dates``     — Reported EPS as fallback (may be standalone)

    Announcement dates always come from ``get_earnings_dates``.
    """

    def fetch_quarterly_eps(
        self, yf_ticker: str, limit: int = _EARNINGS_LIMIT
    ) -> List[Dict]:
        """
        Returns list of ``{date, eps, estimate}`` sorted **oldest-first**,
        using **consolidated** EPS where available.
        """
        print(f"[YFinance] Fetching earnings data for {yf_ticker} …")
        ticker = yf.Ticker(yf_ticker)

        # --- 1. Announcement dates + Reported EPS (fallback) --------------
        df = ticker.get_earnings_dates(limit=limit)
        if df is None or df.empty:
            print(f"[YFinance] No earnings data for {yf_ticker}")
            return []

        # --- 2. Build consolidated EPS lookup (quarter-end → eps) ----------
        consolidated = self._build_consolidated_lookup(ticker)

        # --- 3. Merge: prefer consolidated EPS over Reported EPS -----------
        records: List[Dict] = []
        used_consolidated = 0
        for idx, row in df.iterrows():
            reported = row.get("Reported EPS")
            if pd.isna(reported):
                continue                    # skip future / pending dates

            # Convert tz-aware datetime → naive (keep date & time)
            dt = idx.to_pydatetime()
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)

            qtr_end = self._announcement_to_quarter_end(dt)

            # Prefer consolidated EPS; fall back to Reported EPS
            if qtr_end in consolidated:
                eps = consolidated[qtr_end]
                used_consolidated += 1
            else:
                eps = float(reported)

            estimate = row.get("EPS Estimate")
            records.append({
                "date": dt,
                "eps":  float(eps),
                "estimate": (float(estimate)
                             if not pd.isna(estimate) else None),
            })

        # Sort oldest → newest for TTM computation
        records.sort(key=lambda x: x["date"])
        if records:
            print(f"[YFinance] Got {len(records)} quarterly EPS records  "
                  f"({records[0]['date'].strftime('%b %Y')} → "
                  f"{records[-1]['date'].strftime('%b %Y')})")
            print(f"[YFinance] Consolidated EPS used for "
                  f"{used_consolidated}/{len(records)} quarters")
        return records

    @staticmethod
    def _build_consolidated_lookup(ticker) -> Dict[datetime, float]:
        """
        Build ``{quarter_end_datetime → consolidated_EPS}`` from
        ``earnings_history`` and ``quarterly_income_stmt``.

        ``earnings_history`` values take priority over income-stmt
        computed values because they are company-reported consolidated EPS.
        """
        lookup: Dict[datetime, float] = {}

        # Source A: quarterly_income_stmt  Net Income / sharesOutstanding
        try:
            qi = ticker.quarterly_income_stmt
            shares = ticker.info.get("sharesOutstanding")
            if qi is not None and shares and shares > 0:
                ni_field = None
                for f in ["Net Income Common Stockholders", "Net Income"]:
                    if f in qi.index:
                        ni_field = f
                        break
                if ni_field:
                    for col in qi.columns:
                        ni = qi.loc[ni_field, col]
                        if not pd.isna(ni):
                            qtr_dt = col.to_pydatetime()
                            if qtr_dt.tzinfo:
                                qtr_dt = qtr_dt.replace(tzinfo=None)
                            lookup[qtr_dt] = round(float(ni) / shares, 2)
                    print(f"[YFinance] income_stmt: "
                          f"{len(lookup)} quarters with Net Income")
        except Exception as exc:
            print(f"[YFinance] quarterly_income_stmt error: {exc}")

        # Source B: earnings_history (overrides income-stmt values)
        try:
            eh = ticker.earnings_history
            if eh is not None and not eh.empty:
                count = 0
                for qtr_idx, row in eh.iterrows():
                    eps_actual = row.get("epsActual")
                    if pd.isna(eps_actual):
                        continue
                    qtr_dt = (qtr_idx.to_pydatetime()
                              if hasattr(qtr_idx, "to_pydatetime")
                              else qtr_idx)
                    if hasattr(qtr_dt, "tzinfo") and qtr_dt.tzinfo:
                        qtr_dt = qtr_dt.replace(tzinfo=None)
                    lookup[qtr_dt] = float(eps_actual)
                    count += 1
                print(f"[YFinance] earnings_history: "
                      f"{count} quarters with consolidated EPS")
        except Exception as exc:
            print(f"[YFinance] earnings_history error: {exc}")

        return lookup

    @staticmethod
    def _announcement_to_quarter_end(dt: datetime) -> datetime:
        """Map an earnings announcement date to its fiscal quarter-end date."""
        m, y = dt.month, dt.year
        if 1 <= m <= 3:            # Jan-Mar → Q ending Dec prev year
            return datetime(y - 1, 12, 31)
        elif 4 <= m <= 6:          # Apr-Jun → Q ending Mar
            return datetime(y, 3, 31)
        elif 7 <= m <= 9:          # Jul-Sep → Q ending Jun
            return datetime(y, 6, 30)
        else:                      # Oct-Dec → Q ending Sep
            return datetime(y, 9, 30)


# ---------------------------------------------------------------------------
# Fyers Price Client
# ---------------------------------------------------------------------------
class FyersPriceClient:
    """Fetch daily closing prices from Fyers."""

    def __init__(self):
        FyersService.init(disable_lock=True)
        self.logger = LoggerInitializer.get_instance()

    # ---- bulk daily prices ------------------------------------------------
    def fetch_daily_prices(
        self, fyers_symbol: str, start_date: str, end_date: str
    ) -> List[Tuple[datetime, float]]:
        """
        Fetch daily candles and return ``[(date, close), …]`` sorted ASC.

        Parameters use ``YYYY-MM-DD`` format.  Automatically chunks the
        request by 365-day windows to stay within Fyers API limits.
        """
        print(f"[Fyers] Fetching daily prices for {fyers_symbol}  "
              f"{start_date} → {end_date} …")

        all_candles: List[Tuple[datetime, float]] = []
        dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        dt_end   = datetime.strptime(end_date,   "%Y-%m-%d")

        chunk_days = 365
        current    = dt_start
        chunk_num  = 0

        while current < dt_end:
            chunk_end = min(current + timedelta(days=chunk_days), dt_end)
            chunk_num += 1

            data = {
                "symbol":      fyers_symbol,
                "resolution":  "D",
                "date_format": "1",
                "range_from":  current.strftime("%Y-%m-%d"),
                "range_to":    chunk_end.strftime("%Y-%m-%d"),
            }

            try:
                resp = FyersService.history(data=data, disable_lock=True)
                if resp.get("s") == "ok" and resp.get("candles"):
                    for c in resp["candles"]:
                        dt = datetime.fromtimestamp(c[0])
                        all_candles.append((dt, c[4]))   # (date, close)
                    print(f"  chunk {chunk_num}: "
                          f"{len(resp['candles'])} candles  "
                          f"({current.strftime('%Y-%m-%d')} → "
                          f"{chunk_end.strftime('%Y-%m-%d')})")
                else:
                    print(f"  chunk {chunk_num}: no data  "
                          f"({current.strftime('%Y-%m-%d')} → "
                          f"{chunk_end.strftime('%Y-%m-%d')})")
            except Exception as exc:
                print(f"  chunk {chunk_num}: ERROR {exc}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.3)

        all_candles.sort(key=lambda x: x[0])
        print(f"[Fyers] Total {len(all_candles)} daily candles loaded.\n")
        return all_candles


# ---------------------------------------------------------------------------
# Trailing PE Calculator
# ---------------------------------------------------------------------------
class TrailingPECalculator:
    """
    Orchestrator that computes **Trailing (TTM) PE** with 756-day PE bands.

    Workflow
    --------
    1. Fetch quarterly consolidated EPS  (``EarningsClient``).
    2. Build a **TTM EPS timeline** — at each quarterly earnings date,
       TTM EPS = sum of that quarter + previous 3 quarters.
    3. Fetch daily closing prices from Fyers for the full period.
    4. Assign each trading day the most-recent TTM EPS and compute
       **daily PE = close / TTM EPS**.
    5. For each quarterly result in the report window:
       - **Trailing PE** = close on result date / TTM EPS
       - **PE High / Low / Median** from daily PE values over the
         preceding 756 days.
    """

    def __init__(self):
        self.earnings_client = EarningsClient()
        self.fyers_client    = FyersPriceClient()

    def compute(self, symbol: str, num_quarters: int = 12) -> List[Dict]:
        """
        Compute trailing PE with 756-day bands for the last *num_quarters*.

        Returns list of result dicts (**newest-first**), each with keys:
        ``symbol``, ``earnings_date``, ``quarter_eps``, ``ttm_eps``,
        ``close_price``, ``trailing_pe``,
        ``pe_high_3yr``, ``pe_low_3yr``, ``pe_median_3yr``, ``pe_band_days``.
        """
        mapping   = self._resolve_symbols(symbol)
        yf_ticker = mapping["yf"]
        fyers_sym = mapping["fyers"]

        # ---- 1. Quarterly EPS (oldest-first) ------------------------------
        eps_records = self.earnings_client.fetch_quarterly_eps(
            yf_ticker, limit=_EARNINGS_LIMIT
        )
        if len(eps_records) < 4:
            print(f"[PE] Need ≥ 4 quarters for TTM PE, got {len(eps_records)}")
            return []

        # ---- 2. Build TTM EPS timeline ------------------------------------
        ttm_timeline = self._build_ttm_timeline(eps_records)
        print(f"[PE] TTM EPS timeline: {len(ttm_timeline)} data points  "
              f"({ttm_timeline[0]['date'].strftime('%b %Y')} → "
              f"{ttm_timeline[-1]['date'].strftime('%b %Y')})")

        # ---- 3. Determine which quarters to report ------------------------
        report_count  = min(num_quarters, len(ttm_timeline))
        report_window = ttm_timeline[-report_count:]

        # ---- 4. Fetch daily prices ----------------------------------------
        #    Go back 3 extra years before the earliest report quarter so
        #    the PE-band computation has enough daily data.
        price_start = (report_window[0]["date"]
                       - timedelta(days=_PE_BAND_DAYS + 60))
        price_end   = datetime.now()

        daily_prices = self.fyers_client.fetch_daily_prices(
            fyers_sym,
            price_start.strftime("%Y-%m-%d"),
            price_end.strftime("%Y-%m-%d"),
        )
        if not daily_prices:
            print("[PE] No daily price data available from Fyers")
            return []

        # ---- 5. Daily PE series -------------------------------------------
        daily_pe = self._compute_daily_pe(daily_prices, ttm_timeline)
        print(f"[PE] Computed {len(daily_pe)} daily PE values\n")

        # ---- 6. Build results for each quarter ----------------------------
        results: List[Dict] = []
        for i, rec in enumerate(report_window):
            row = self._quarter_result(symbol, rec, daily_prices, daily_pe)
            results.append(row)

            # Progress log
            pe_str   = f"{row['trailing_pe']:.2f}" if row["trailing_pe"] else "N/A"
            hi_str   = f"{row['pe_high_3yr']:.2f}" if row["pe_high_3yr"] else "N/A"
            lo_str   = f"{row['pe_low_3yr']:.2f}"  if row["pe_low_3yr"]  else "N/A"
            med_str  = f"{row['pe_median_3yr']:.2f}" if row["pe_median_3yr"] else "N/A"
            print(f"  [{i+1}/{report_count}]  "
                  f"{row['earnings_date']:<14}  "
                  f"Q_EPS={row['quarter_eps']:>7.2f}  "
                  f"TTM={row['ttm_eps']:>8.2f}  "
                  f"PE={pe_str:>8}  "
                  f"Hi={hi_str:>8}  Lo={lo_str:>8}  Med={med_str:>8}")

        # Return newest-first
        results.reverse()

        # ---- 7. Current PE (last trading session close) ------------------
        current_entry = self._current_pe(
            symbol, daily_prices, daily_pe, ttm_timeline
        )
        if current_entry:
            # Only prepend if the current date differs from the newest quarter
            newest_q_date = results[0]["earnings_date"] if results else ""
            if current_entry["earnings_date"] != newest_q_date:
                results.insert(0, current_entry)
                pe_s = (f"{current_entry['trailing_pe']:.2f}"
                        if current_entry['trailing_pe'] else 'N/A')
                print(f"  [CURRENT]  "
                      f"{current_entry['earnings_date']:<14}  "
                      f"TTM={current_entry['ttm_eps']:>8.2f}  "
                      f"Price={current_entry['close_price']:>10,.2f}  "
                      f"PE={pe_s:>8}")

        return results

    # ---- internal helpers -------------------------------------------------

    @staticmethod
    def _build_ttm_timeline(eps_records: List[Dict]) -> List[Dict]:
        """
        From sorted (oldest-first) quarterly EPS records, compute
        TTM EPS at each quarter (starting from the 4th).

        Returns ``[{date, eps, ttm_eps}, …]`` sorted oldest-first.
        """
        timeline: List[Dict] = []
        for i in range(3, len(eps_records)):
            q_eps = [eps_records[j]["eps"] for j in range(i - 3, i + 1)]
            timeline.append({
                "date":    eps_records[i]["date"],
                "eps":     eps_records[i]["eps"],
                "ttm_eps": sum(q_eps),
            })
        return timeline

    @staticmethod
    def _compute_daily_pe(
        daily_prices: List[Tuple[datetime, float]],
        ttm_timeline: List[Dict],
    ) -> List[Tuple[datetime, float]]:
        """
        For each daily close, assign the most-recent TTM EPS effective on
        that date and compute ``PE = close / TTM EPS``.

        Skips days where TTM EPS ≤ 0 (loss-making periods — PE undefined).
        Returns sorted ``[(date, pe), …]``.
        """
        daily_pe: List[Tuple[datetime, float]] = []
        ttm_idx = 0

        for dt, close in daily_prices:
            # Advance to the latest TTM EPS announced on or before 'dt'
            while (ttm_idx < len(ttm_timeline) - 1
                   and ttm_timeline[ttm_idx + 1]["date"] <= dt):
                ttm_idx += 1

            if (ttm_idx < len(ttm_timeline)
                    and ttm_timeline[ttm_idx]["date"] <= dt
                    and ttm_timeline[ttm_idx]["ttm_eps"] > 0):
                pe = close / ttm_timeline[ttm_idx]["ttm_eps"]
                daily_pe.append((dt, pe))

        return daily_pe

    @staticmethod
    def _find_closest_price(
        daily_prices: List[Tuple[datetime, float]], target: datetime
    ) -> Optional[float]:
        """Return close price on the trading day closest to *target* (±7 d)."""
        if not daily_prices:
            return None
        best = min(daily_prices,
                   key=lambda x: abs((x[0] - target).total_seconds()))
        if abs((best[0] - target).days) <= 7:
            return best[1]
        return None

    def _quarter_result(
        self,
        symbol: str,
        rec: Dict,
        daily_prices: List[Tuple[datetime, float]],
        daily_pe: List[Tuple[datetime, float]],
    ) -> Dict:
        """Build the output dict for one quarterly earnings announcement."""
        earnings_date = rec["date"]
        ttm_eps       = rec["ttm_eps"]
        quarter_eps   = rec["eps"]

        # Price on the earnings-announcement date
        close_price = self._find_closest_price(daily_prices, earnings_date)

        # Trailing PE
        trailing_pe = None
        if close_price and ttm_eps and ttm_eps > 0:
            trailing_pe = round(close_price / ttm_eps, 2)

        # 756-day PE bands (from daily PE values)
        cutoff = earnings_date - timedelta(days=_PE_BAND_DAYS)
        window_pe = [pe for dt, pe in daily_pe
                     if cutoff <= dt <= earnings_date]

        pe_high   = round(max(window_pe), 2) if window_pe else None
        pe_low    = round(min(window_pe), 2) if window_pe else None
        pe_median = round(statistics.median(window_pe), 2) if window_pe else None

        return {
            "symbol":         symbol,
            "earnings_date":  earnings_date.strftime("%d %b %Y"),
            "quarter_eps":    round(quarter_eps, 2),
            "ttm_eps":        round(ttm_eps, 2),
            "close_price":    round(close_price, 2) if close_price else None,
            "trailing_pe":    trailing_pe,
            "pe_high_3yr":    pe_high,
            "pe_low_3yr":     pe_low,
            "pe_median_3yr":  pe_median,
            "pe_band_days":   len(window_pe),
        }

    def _current_pe(
        self,
        symbol: str,
        daily_prices: List[Tuple[datetime, float]],
        daily_pe: List[Tuple[datetime, float]],
        ttm_timeline: List[Dict],
    ) -> Optional[Dict]:
        """Build a 'Current' PE entry using the last trading session close."""
        if not daily_prices or not ttm_timeline:
            return None

        last_date, last_close = daily_prices[-1]
        latest_ttm = ttm_timeline[-1]
        ttm_eps    = latest_ttm["ttm_eps"]

        trailing_pe = None
        if ttm_eps and ttm_eps > 0:
            trailing_pe = round(last_close / ttm_eps, 2)

        # 756-day PE bands up to today
        cutoff = last_date - timedelta(days=_PE_BAND_DAYS)
        window_pe = [pe for dt, pe in daily_pe if cutoff <= dt <= last_date]

        pe_high   = round(max(window_pe), 2) if window_pe else None
        pe_low    = round(min(window_pe), 2) if window_pe else None
        pe_median = (round(statistics.median(window_pe), 2)
                     if window_pe else None)

        return {
            "symbol":         symbol,
            "earnings_date":  f"Current ({last_date.strftime('%d %b %Y')})",
            "quarter_eps":    round(latest_ttm["eps"], 2),
            "ttm_eps":        round(ttm_eps, 2),
            "close_price":    round(last_close, 2),
            "trailing_pe":    trailing_pe,
            "pe_high_3yr":    pe_high,
            "pe_low_3yr":     pe_low,
            "pe_median_3yr":  pe_median,
            "pe_band_days":   len(window_pe),
        }

    @staticmethod
    def _resolve_symbols(symbol: str) -> Dict[str, str]:
        sym = symbol.upper()
        if sym in SYMBOL_MAP:
            return SYMBOL_MAP[sym]
        return {"yf": f"{sym}.NS", "fyers": f"NSE:{sym}-EQ"}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_results(results: List[Dict], symbol: str, output_dir: Path):
    """Persist results as a pretty-printed JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{symbol}_pe_history.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅  Saved {len(results)} records → {filepath}")


def print_summary_table(results: List[Dict]):
    """Pretty-print a trailing-PE summary table to the console."""
    if not results:
        print("\nNo results to display.\n")
        return

    sym = results[0].get("symbol", "?")
    hdr = (f"  {'Result Date':<26} {'Q EPS':>7} {'TTM EPS':>8} "
           f"{'Price':>10} {'TTM PE':>8}  │ "
           f"{'Hi(756d)':>9} {'Lo(756d)':>9} {'Med(756d)':>10}")

    print(f"\n{'=' * 100}")
    print(f"  Trailing PE History — {sym}   "
          f"(TTM PE = Price / last 4 quarters EPS)")
    print(f"{'=' * 100}")
    print(hdr)
    print(f"  {'-' * 94}")

    for r in results:
        date_s  = r.get("earnings_date", "")
        is_current = date_s.startswith("Current")
        label   = date_s if not is_current else date_s  # keep as-is
        q_eps   = f"{r['quarter_eps']:>7.2f}"
        ttm     = f"{r['ttm_eps']:>8.2f}"
        price   = (f"{r['close_price']:>10,.2f}"
                   if r["close_price"] else f"{'N/A':>10}")
        pe      = (f"{r['trailing_pe']:>8.2f}"
                   if r["trailing_pe"] else f"{'N/A':>8}")
        hi      = (f"{r['pe_high_3yr']:>8.2f}"
                   if r["pe_high_3yr"] else f"{'N/A':>8}")
        lo      = (f"{r['pe_low_3yr']:>8.2f}"
                   if r["pe_low_3yr"] else f"{'N/A':>8}")
        med     = (f"{r['pe_median_3yr']:>8.2f}"
                   if r["pe_median_3yr"] else f"{'N/A':>8}")

        row_text = (f"  {label:<26} {q_eps} {ttm} {price} {pe}  │ "
                    f"{hi} {lo} {med}")
        if is_current:
            print(f"▶{row_text[1:]}  ◀")
            print(f"  {'-' * 94}")
        else:
            print(row_text)

    print(f"  {'=' * 94}")

    # Current PE highlight
    current = results[0] if results else {}
    if (current.get("earnings_date", "").startswith("Current")
            and current.get("trailing_pe")):
        print(f"\n  🔴 Current PE : {current['trailing_pe']:.2f}  "
              f"(Price ₹{current['close_price']:,.2f}  /  "
              f"TTM EPS ₹{current['ttm_eps']:.2f})")

    # Quick stats from the quarterly trailing PEs (exclude Current row)
    pe_vals = [r["trailing_pe"] for r in results
               if r["trailing_pe"] is not None
               and not r.get("earnings_date", "").startswith("Current")]
    if pe_vals:
        print(f"\n  📊 Trailing PE Statistics  ({len(pe_vals)} quarters)")
        print(f"     Latest TTM PE : {pe_vals[0]:.2f}")
        print(f"     Min TTM PE    : {min(pe_vals):.2f}")
        print(f"     Max TTM PE    : {max(pe_vals):.2f}")
        print(f"     Avg TTM PE    : {sum(pe_vals)/len(pe_vals):.2f}")

    # Latest 756-day band
    latest = results[0] if results else {}
    if latest.get("pe_high_3yr"):
        print(f"\n  📈 Current 756-Day PE Band  "
              f"({latest.get('pe_band_days', 0)} trading days)")
        print(f"     PE High   : {latest['pe_high_3yr']:.2f}")
        print(f"     PE Median : {latest['pe_median_3yr']:.2f}")
        print(f"     PE Low    : {latest['pe_low_3yr']:.2f}")
    print()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def load_symbols(filepath: Path) -> List[str]:
    """Read one symbol per line from a text file (ignoring blanks / #comments)."""
    if not filepath.exists():
        print(f"Symbol file not found: {filepath}")
        return []

    symbols: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                symbols.append(line.upper())
    print(f"[Batch] Loaded {len(symbols)} symbols from {filepath}")
    return symbols


def run_batch(symbols: List[str], num_quarters: int):
    """Process multiple symbols sequentially, saving each to output/."""
    calc = TrailingPECalculator()
    all_summaries: Dict = {}

    for i, sym in enumerate(symbols):
        print(f"\n{'#' * 70}")
        print(f"  [{i+1}/{len(symbols)}]  Processing {sym}")
        print(f"{'#' * 70}")

        results = calc.compute(sym, num_quarters)
        print_summary_table(results)
        save_results(results, sym, OUTPUT_DIR)

        pe_vals = [r["trailing_pe"] for r in results
                   if r["trailing_pe"] is not None]
        all_summaries[sym] = {
            "records":   len(results),
            "min_pe":    min(pe_vals)   if pe_vals else None,
            "max_pe":    max(pe_vals)   if pe_vals else None,
            "avg_pe":    (round(sum(pe_vals) / len(pe_vals), 2)
                          if pe_vals else None),
            "latest_pe": pe_vals[0]     if pe_vals else None,
        }

        if i < len(symbols) - 1:
            print("  ⏳ Waiting 1 s before next symbol …")
            time.sleep(1)

    # Save combined summary
    summary_path = OUTPUT_DIR / "_batch_summary.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✅  Batch summary saved → {summary_path}\n")

    # Consolidated table
    print(f"\n{'=' * 68}")
    print(f"  Batch Summary  (Trailing TTM PE)")
    print(f"{'=' * 68}")
    print(f"  {'Symbol':<12} {'Records':>8} "
          f"{'Min PE':>8} {'Max PE':>8} {'Avg PE':>8} {'Latest':>8}")
    print(f"  {'-' * 60}")
    for sym, s in all_summaries.items():
        _f = lambda v: f"{v:.2f}" if v is not None else "N/A"
        print(f"  {sym:<12} {s['records']:>8} "
              f"{_f(s['min_pe']):>8} {_f(s['max_pe']):>8} "
              f"{_f(s['avg_pe']):>8} {_f(s['latest_pe']):>8}")
    print(f"  {'=' * 60}\n")


# ---------------------------------------------------------------------------
# Single-symbol run
# ---------------------------------------------------------------------------
def run_single(symbol: str, num_quarters: int):
    """Process one symbol and save results."""
    mapping = SYMBOL_MAP.get(
        symbol, {"yf": f"{symbol}.NS", "fyers": f"NSE:{symbol}-EQ"})

    print(f"\n{'=' * 60}")
    print(f"  Trailing PE Calculator — {symbol}")
    print(f"  YFinance      : {mapping['yf']}")
    print(f"  Fyers symbol  : {mapping['fyers']}")
    print(f"  Quarters      : {num_quarters}")
    print(f"  PE Band       : {_PE_BAND_DAYS}-day lookback (daily PE)")
    print(f"{'=' * 60}\n")

    calc    = TrailingPECalculator()
    results = calc.compute(symbol, num_quarters)
    print_summary_table(results)
    save_results(results, symbol, OUTPUT_DIR)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Trailing TTM PE Calculator with 756-Day PE Bands  "
                    "(Consolidated EPS + Fyers Prices)"
    )
    parser.add_argument(
        "--symbol", type=str, default="RELIANCE",
        help="Short symbol name (e.g. SBIN, RELIANCE). Default: RELIANCE",
    )
    parser.add_argument(
        "--quarters", type=int, default=12,
        help="Number of recent quarters to show. Default: 12",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Batch mode: process all symbols listed in the symbols file.",
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Path to symbols text file (one symbol per line). "
             f"Default: {DEFAULT_SYMBOLS_FILE}",
    )
    args = parser.parse_args()

    if args.batch:
        symbols_file = Path(args.file) if args.file else DEFAULT_SYMBOLS_FILE
        symbols = load_symbols(symbols_file)
        if not symbols:
            print("No symbols to process. Exiting.")
            return
        run_batch(symbols, args.quarters)
    else:
        run_single(args.symbol.upper(), args.quarters)


if __name__ == "__main__":
    main()
