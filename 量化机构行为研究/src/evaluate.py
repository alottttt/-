# -*- coding: utf-8 -*-
"""只读展示策略对照；原独立 eval_rolling 产出已退役，不再生成机构排名表。"""
import argparse
import sys

from strategy import load_results, print_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="全期", help="全期或年份，例如 2025")
    args = parser.parse_args(argv)
    _, metrics, config = load_results()
    if args.period not in set(metrics["期间"]):
        parser.error(f"不存在期间 {args.period}")
    print(f"策略批次 {config['run_id']}，行情截至 {config['as_of']}")
    print_comparison(metrics, args.period)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
