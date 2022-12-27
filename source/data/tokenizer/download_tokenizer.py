import pathlib
from tokenizer import path_tokenizer_ressources
import argparse
from transformers import BertTokenizerFast

def main(args : argparse.Namespace) -> None :


    tokenizer = BertTokenizerFast.from_pretrained(args.key)

    # creating directory
    destination_path = path_tokenizer_ressources / args.name
    destination_path.mkdir(parents=True, exist_ok=True)
    print(f"creating directory : {destination_path}")
    tokenizer.save_pretrained(str(destination_path))

    print("done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("key", type = str)
    parser.add_argument("name", type = str)

    args = parser.parse_args()

    main(args)