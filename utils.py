"""Dataset-specific natural-language prompts."""

from __future__ import annotations


PROMPTS = {
    "Toy": (
        "Given a list of toys the user has played with, recommend a new toy the user may like.",
        "The user has played with the following toys: ",
    ),
    "Amazon_Books": (
        "Given a list of books the user has read, recommend a new book the user may like.",
        "The user has read the following books: ",
    ),
    "Books": (
        "Given a list of books the user has read, recommend a new book the user may like.",
        "The user has read the following books: ",
    ),
    "Clothing": (
        "Given a list of clothing items the user has worn, recommend a new clothing item the user may like.",
        "The user has worn the following clothing items: ",
    ),
    "Office": (
        "Given a list of office products the user has used, recommend a new office product the user may like.",
        "The user has used the following office products: ",
    ),
    "Sports": (
        "Given a list of sports products the user has used, recommend a new sports product the user may like.",
        "The user has used the following sports products: ",
    ),
    "Movielens": (
        "Given a list of movies the user has watched, recommend a new movie the user may like.",
        "The user has watched the following movies: ",
    ),
    "Beauty": (
        "Given a list of beauty products the user has used, recommend a new beauty product the user may like.",
        "The user has used the following beauty products: ",
    ),
    "Yelp": (
        "Given a list of businesses the user has visited, recommend a new business the user may like.",
        "The user has visited the following businesses: ",
    ),
    "Steam": (
        "Given a list of games the user has played, recommend a new game the user may like.",
        "The user has played the following games: ",
    ),
    "CD": (
        "Given a list of music products the user has purchased, recommend a new music product the user may like.",
        "The user has purchased the following music products: ",
    ),
}


def get_prompt(dataset_name: str) -> tuple[str, str]:
    for key, prompt in PROMPTS.items():
        if key.lower() in dataset_name.lower():
            return prompt
    return (
        "Given the user's interaction history, recommend a new item the user may like.",
        "The user has interacted with the following items: ",
    )
