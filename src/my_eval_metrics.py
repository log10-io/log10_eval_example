import re
from typing import List
from openai import OpenAI
from scipy import spatial

client = OpenAI()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_distance(refence_text: str, text: str) -> float:
    ref_embedding = get_embedding(refence_text)
    text_embedding = get_embedding(text)
    cosine_distance = spatial.distance.cosine(ref_embedding, text_embedding)
    return cosine_distance


def count_words(text: str) -> int:
    # Remove symbols and keep only alphanumeric characters and spaces
    clean_string = re.sub(r"[^\w\s]", "", text)
    # Split the string into words and count the number of words
    word_count = len(clean_string.split())
    return word_count


def main():
    text_1 = "Once upon a time, in a land far, far away, there was a little"
    text_2 = "Once upon a time, in a land far, far away, there was"

    print(cosine_distance(text_1, text_2))


if __name__ == "__main__":
    main()
