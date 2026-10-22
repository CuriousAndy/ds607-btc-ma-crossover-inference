"""Pull and load Binance daily candles. Owner: Deng Shenbojia (A1)."""
from datetime import datetime, timezone  # timestamps for the API and the pull log

import pandas as pd  # tables, CSV read/write, date handling
import requests  # HTTP calls to the Binance REST API

from src import config  # pre-registered parameters (symbol, dates, folders)

KLINES_URL = "https://api.binance.com/api/v3/klines"
KLINES_URL_FALLBACK = "https://data-api.binance.vision/api/v3/klines"

# The 12 fields Binance returns for every candle, in order (Binance API docs, "Kline/Candlestick data")
KLINE_COLUMNS = [
    "open_time",  # candle open, milliseconds since 1970-01-01 UTC
    "open",
    "high",
    "low",
    "close",
    "volume",  # base asset volume (BTC)
    "close_time",  # candle close, ms UTC (23:59:59.999 of the same UTC day)
    "quote_asset_volume",  # volume in USDT
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",  # unused field, kept so the raw file has all 12 columns
]

LIMIT = 1000  # Binance returns at most 1,000 candles per request
DAY_MS = 24 * 60 * 60 * 1000  # one day in milliseconds


def _to_ms(day):
    """'YYYY-MM-DD' -> milliseconds at 00:00 UTC of that day."""
    dt = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)  # midnight UTC
    return int(dt.timestamp() * 1000)  # Binance wants integer milliseconds


def raw_path(start=config.PULL_START, end=config.SAMPLE_END):
    """data/raw/btcusdt_1d_YYYYMMDD_YYYYMMDD.csv for the given pull window."""
    name = f"{config.SYMBOL.lower()}_{config.INTERVAL}_{start.replace('-', '')}_{end.replace('-', '')}.csv"
    return config.RAW_DIR / name  # e.g. data/raw/btcusdt_1d_20181201_20251231.csv


def _get_page(params):
    """One request; try the main endpoint, then the fallback. Returns (rows, full_url)."""
    last_error = None  # remember why the main endpoint failed
    for url in (KLINES_URL, KLINES_URL_FALLBACK):  # same format on both endpoints
        try:
            resp = requests.get(url, params=params, timeout=30)  # one HTTP GET
            resp.raise_for_status()  # turn 4xx/5xx into an exception
            return resp.json(), resp.url  # list of candles + the exact URL with parameters
        except requests.RequestException as err:  # connection refused, blocked region, HTTP error
            last_error = err  # try the next endpoint
    raise RuntimeError(f"Both Binance endpoints failed: {last_error}")


def fetch_klines(start=config.PULL_START, end=config.SAMPLE_END):
    """Page through the klines endpoint (limit=1000), save all 12 columns to data/raw/ and
    write data/raw/PULL_LOG.md (pull time UTC, URL, parameters, row count)."""
    start_ms = _to_ms(start)  # open time of the first candle we want
    end_ms = _to_ms(end) + DAY_MS - 1  # 23:59:59.999 UTC on the last day, so that day is included
    pulled_at = datetime.now(timezone.utc)  # pull time for the log

    rows, urls = [], []  # all candles, and every request URL for the log
    cursor = start_ms  # open time where the next page starts
    while cursor <= end_ms:  # keep paging until we pass the end date
        params = {
            "symbol": config.SYMBOL,  # BTCUSDT spot
            "interval": config.INTERVAL,  # 1d candles, cut at 00:00 UTC
            "startTime": cursor,
            "endTime": end_ms,
            "limit": LIMIT,
        }
        page, url = _get_page(params)  # up to 1,000 candles
        urls.append(url)  # log the exact request
        if not page:  # nothing left in the window
            break
        rows.extend(page)  # add this page's candles
        cursor = int(page[-1][0]) + DAY_MS  # next page starts the day after the last candle
        if len(page) < LIMIT:  # a short page means we reached the end
            break

    if not rows:  # API answered but sent no candles: stop with a clear message
        raise RuntimeError(f"Binance returned no candles for {start} to {end}; see {urls[0]}")
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)  # 12 columns exactly as Binance sent them
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")  # guard against page overlap

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)  # make sure data/raw/ exists
    path = raw_path(start, end)
    df.to_csv(path, index=False)  # raw file, committed to the repo

    first_day = pd.to_datetime(df["open_time"].iloc[0], unit="ms", utc=True).date()  # for the log
    last_day = pd.to_datetime(df["open_time"].iloc[-1], unit="ms", utc=True).date()
    log_lines = [
        "# Pull log",
        "",
        f"- Pulled at (UTC): {pulled_at:%Y-%m-%d %H:%M:%S}",
        f"- Endpoint: {urls[0].split('?')[0]}",
        f"- Parameters: symbol={config.SYMBOL}, interval={config.INTERVAL}, "
        f"startTime={start_ms} ({start} 00:00 UTC), endTime={end_ms} ({end} 23:59:59.999 UTC), limit={LIMIT}",
        f"- Requests ({len(urls)} pages, full URLs):",
        *[f"  - {u}" for u in urls],
        f"- Rows saved: {len(df)}",
        f"- First candle open (UTC date): {first_day}",
        f"- Last candle open (UTC date): {last_day}",
        f"- File: data/raw/{path.name}",
        "",
    ]
    (config.RAW_DIR / "PULL_LOG.md").write_text("\n".join(log_lines), encoding="utf-8")  # reproducibility record
    print(f"Saved {len(df)} rows to {path.name} ({len(urls)} requests)")
    return df


def load_daily():
    """Read the raw CSV (pull it first if missing). Asserts: N_PULL rows, no duplicate or
    missing days, close > 0. Column `date` is the UTC date of the candle open."""
    path = raw_path()  # data/raw/btcusdt_1d_20181201_20251231.csv
    if not path.is_file():  # first run on a new computer
        fetch_klines()  # pull and save it
    df = pd.read_csv(path)  # all 12 raw columns

    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None).dt.normalize()  # UTC day of the open
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = df[col].astype(float)  # prices and volumes as numbers
    df = df.sort_values("date").reset_index(drop=True)  # oldest first
    df = df[["date"] + KLINE_COLUMNS]  # date first, then the 12 raw columns

    # Data checks: stop immediately if the file is not what the plan expects
    assert len(df) == config.N_PULL, f"expected {config.N_PULL} rows, got {len(df)}"
    assert df["date"].is_unique, "duplicate days in the raw file"
    gaps = df["date"].diff().dropna()  # time between consecutive candles
    assert (gaps == pd.Timedelta(days=1)).all(), "missing days in the raw file"
    assert df["date"].iloc[0] == pd.Timestamp(config.PULL_START), "first day is not PULL_START"
    assert df["date"].iloc[-1] == pd.Timestamp(config.SAMPLE_END), "last day is not SAMPLE_END"
    assert (df["close"] > 0).all(), "non-positive close price"
    return df
