from __future__ import annotations
import argparse, yaml, time, random
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from rich import print

from app import scan_once, save_reports, print_table, with_lock

def job(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    lockfile = cfg.get('scheduler', {}).get('lockfile', '.scan.lock')
    try:
        with with_lock(lockfile):
            rows = scan_once(cfg)
            print_table(rows)
            save_reports(rows, cfg['report']['out_dir'])
    except RuntimeError as e:
        print(f"[yellow]{e} – pomijam ten cykl.[/yellow]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    interval = int(cfg.get('scheduler', {}).get('interval_minutes', 3))
    jitter = int(cfg.get('scheduler', {}).get('jitter_seconds', 10))
    print(f"[green]APScheduler: co {interval} min (jitter {jitter}s). Ctrl+C aby zakończyć.[/green]")

    sched = BlockingScheduler(timezone="UTC")
    trigger = IntervalTrigger(minutes=interval, jitter=jitter)
    sched.add_job(job, trigger, args=[args.config], id="scan_job", max_instances=1, coalesce=True)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("[red]Scheduler zatrzymany.[/red]")

if __name__ == '__main__':
    main()
