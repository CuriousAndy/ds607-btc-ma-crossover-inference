"""Returns, moving averages, lagged signal and excess return. Owner: Deng Shenbojia (A1)."""
import numpy as np  # log, log1p
import pandas as pd  # time-series operations (shift, rolling)

from src import config  # pre-registered parameters (SHORT, LONG, COST, window)


def build_series(df):
    """Return the N_SAMPLE rows of the analysis window with columns:
    r, sma5, sma20, s (signal from day t-1, i.e. shift(1)), r_rule, d, switch, d_cost.
    Checks: no missing s in the window; d == 0 whenever s == 1.

    `close` is also kept (first column) so notebook 01 can plot price against the SMAs."""
    x = df.set_index("date").sort_index()  # one row per UTC day, oldest first
    close = x["close"]  # daily close C_t (UTC 23:59:59)

    r = np.log(close / close.shift(1))  # buy-and-hold log return r_t = ln(C_t / C_{t-1})
    sma5 = close.rolling(config.SHORT).mean()  # mean of closes t-4..t (uses day t's close)
    sma20 = close.rolling(config.LONG).mean()  # mean of closes t-19..t

    above = (sma5 > sma20).astype(float).where(sma20.notna())  # 1.0 if SMA5 > SMA20 on day t; NaN during warm-up
    s = above.shift(1)  # KEY STEP: position for day t uses the SMAs of day t-1 (no look-ahead)

    r_rule = s * r  # in BTC (s=1) earn r_t; in USDT (s=0) earn 0
    d = r_rule - r  # excess return over buy-and-hold; equals -(1 - s) r_t
    switch = (s != s.shift(1)).astype(int)  # 1 on a day the position changes
    d_cost = d + switch * np.log1p(-config.COST)  # charge ln(1 - 0.1%) on every switch day

    out = pd.DataFrame(
        {
            "close": close,
            "r": r,
            "sma5": sma5,
            "sma20": sma20,
            "s": s,
            "r_rule": r_rule,
            "d": d,
            "switch": switch,
            "d_cost": d_cost,
        }
    )
    out = out.loc[config.SAMPLE_START : config.SAMPLE_END].copy()  # drop the December 2018 warm-up

    # Self-checks: stop immediately if the series is not what the plan expects
    assert len(out) == config.N_SAMPLE, f"expected {config.N_SAMPLE} rows, got {len(out)}"
    assert out["s"].notna().all(), "missing signal inside the analysis window"
    assert out["r"].notna().all(), "missing return inside the analysis window"
    assert set(out["s"].unique()) <= {0.0, 1.0}, "signal must be 0 or 1"
    assert (out.loc[out["s"] == 1, "d"] == 0).all(), "d must be exactly 0 on days in BTC"
    out["s"] = out["s"].astype(int)  # 0/1 integers after the checks pass
    return out


def summary(series):
    """Numbers to compare with the trial run (task pack, A1)."""
    return {
        "n": len(series),  # trial: 2,557
        "first_day": series.index[0].date(),
        "last_day": series.index[-1].date(),
        "share_in_btc": series["s"].mean(),  # trial: 55.4%
        "n_switches": int(series["switch"].sum()),  # trial: 143
        "d_bar": series["d"].mean(),  # trial: -0.00021
        "d_bar_annual": series["d"].mean() * 365,  # trial: about -7.6% (log)
        "cum_rule": series["r_rule"].sum(),  # trial: 2.63
        "cum_buy_hold": series["r"].sum(),  # trial: 3.16
    }


if __name__ == "__main__":  # `python -m src.strategy` prints the A1 check numbers
    from src.data import load_daily

    for key, value in summary(build_series(load_daily())).items():
        print(f"{key:>14}: {value:.6f}" if isinstance(value, float) else f"{key:>14}: {value}")
