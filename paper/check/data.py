import argparse
import csv


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dev_set_location", help="where aesw-dev.csv is",
    )

    args = parser.parse_args()

    total_pos = 0

    with open(args.dev_set_location) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # headers

        for _, _, label_str in reader:
            total_pos += int(label_str)

    assert total_pos == 57340

    print("Validation data is good! :)")


if __name__ == "__main__":
    main()
