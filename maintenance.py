# maintenance.py
def job_train_meta():
    try:
        import train_meta
        train_meta.main()
    except Exception as e:
        print(f"[AUTO][TRAIN] error: {e}")

def job_calibrate_meta():
    try:
        import calibrate_meta
        calibrate_meta.main()
    except Exception as e:
        print(f"[AUTO][CALIB] error: {e}")

def job_tune_thresholds(mode=None, window_days=None):
    try:
        import tune_thresholds
        # wersja programatyczna (patrz plik tune_thresholds.py)
        tune_thresholds.tune_thresholds_programmatic(mode=mode, window_days=window_days)
    except Exception as e:
        print(f"[AUTO][TUNE] error: {e}")
