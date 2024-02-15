from openai import OpenAI
from scipy import spatial

client = OpenAI()

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(refence_text, text):
    ref_embedding = get_embedding(refence_text)
    text_embedding = get_embedding(text)
    cosine_similarity = spatial.distance.cosine(ref_embedding, text_embedding)
    return cosine_similarity

def main():
    text_1 = "Once upon a time, in a land far, far away, there was a little"
    text_2 = "Once upon a time, in a land far, far away, there was"

    print(cosine_similarity(text_1, text_2))

if __name__ == "__main__":
    main()