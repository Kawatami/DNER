import string
from typing import Callable

def lower_case(text : str) -> str :
    """
    Convert text to ist lower case version
    :param text: text to process
    :return: lower cased text
    """

    return text.lower()

def remove_punctuation(text : str) -> str :
    """
    Convert text to ist lower case version
    :param text: text to process
    :return: lower cased text
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_first_sentence(text : str) -> str :
    """
    Remove the first sentence of text
    :param text: input text
    :return: updated text
    """
    text = text.split(".")[1:-1]
    return ".".join(text)


def resolve_text_preprocessing(preproc : str) -> Callable :
    """
    Convert a string to its corresponding function
    :param preproc: string representation of the preprocessing
    :return: associated function
    """

    return text_preprocessing[preproc]

text_preprocessing = {
    "lower_case": lower_case,
    "remove_punctuation" : remove_punctuation,
    "remove_first_sentence" : remove_first_sentence
}



