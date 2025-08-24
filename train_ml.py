from __future__ import annotations
import argparse, yaml
from rich import print
from core.ml import train_and_save

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Wymuś 2 lata jeśli w configu ustawiono days:null i years:2
    ml_cfg = cfg.get('ml', {})
    if ml_cfg.get('days', None) is None:
        print("[cyan]Trenuję model ML na okresie wg 'years' (domyślnie 2 lata)...[/cyan]")
    else:
        print(f"[cyan]Trenuję model ML na ostatnich {ml_cfg['days']} dniach...[/cyan]")

    meta = train_and_save(cfg)
    if 'warning' in meta:
        print(f"[yellow]{meta['warning']}[/yellow]")
    print("[green]Zapisano model i metadane.[/green]")
    print(meta)

if __name__ == '__main__':
    main()
