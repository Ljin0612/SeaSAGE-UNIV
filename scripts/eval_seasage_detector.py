#!/usr/bin/env python
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', default='test')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--project', default='runs/eval')
    p.add_argument('--name', default='exp')
    p.parse_args()
    raise NotImplementedError(
        "Evaluation is not implemented yet. Do not record fake zero metrics."
    )


if __name__ == '__main__':
    main()
