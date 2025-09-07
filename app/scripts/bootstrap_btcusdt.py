"""
Bootstrap script for BTCUSDT on Bitget using the shared pipeline:
- Runs the same logic as the API /scan, prints results to stdout.
"""
from app.pipeline.scan import scan_symbols

def main() -> None:
    signals = scan_symbols(
        symbols=["BTCUSDT"],
        equity=5_000.0,
        risk_profile="medium",
        signal_tfs=("10m", "15m", "30m"),
        run_ingest=True,
    )

    print("\n=== SIGNALS (BTCUSDT) ===")
    if not signals:
        print("No candidates passed HTF filter in the latest window.")
    else:
        for s in signals:
            print(s)

if __name__ == "__main__":
    main()
