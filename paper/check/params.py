import os

from ..structures import HyperParameters


def main() -> None:
    count = 0
    for foldername in os.listdir("experiments"):
        params = os.path.join("experiments", foldername, "params.toml")

        if not os.path.isfile(params):
            continue

        try:
            HyperParameters(params)
        except KeyError as err:
            print(params, "is missing", str(err))
            count += 1
    if count == 0:
        print("All param files are good! :)")
    else:
        print(f"{count} param file(s) missing keys.")


if __name__ == "__main__":
    main()
