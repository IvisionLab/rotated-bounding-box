#!/usr/bin/env python3
import sys
from maskrcnn import anns

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate a json file containing annotations")

    parser.add_argument("dataset_folder",
                        help="The path to a folder with images and annotations")

    parser.add_argument("--output",
                        default="annotations.json",
                        help="The output filepath")

    parser.add_argument("--split",
                        type=bool,
                        default=False,
                        help="Split dataset into train and validation")

    parser.add_argument("--limit",
                        type=int,
                        help="Maximum number of images per classes")

    parser.add_argument("--use-mask",
                        const=True,
                        nargs='?',
                        type=bool,
                        metavar="<True/False>",
                        default=False,
                        help="Save annotation mask")

    args = parser.parse_args()

    anns.rbox.generate(args)
