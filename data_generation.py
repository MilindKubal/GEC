import pandas as pd
from tqdm import tqdm
import pickle
# df = pd.read_csv(r"D:\GEC\Amazon reviews\amazon_reviews_us_Software_v1_00.tsv\amazon_reviews_us_Software_v1_00.tsv",  sep='\t', error_bad_lines=False)
#
# df.to_csv("amazon_software_reviews.csv")


def generate_corr_sentences(data):
    data = pd.read_csv(data,  sep='\t', error_bad_lines=False)

    reviews = data['review_body'].tolist()

    with open("corr_sentences_train.txt","w",encoding="utf-8") as files:

        for review in tqdm(reviews):
            res = str(review).split(".")
            if len(res) < 3:
                files.write(str(review).replace('<br />',"")+"\n")
            else:
                pass


# generate_corr_sentences(data = r"D:\GEC\Amazon reviews\amazon_reviews_us_Software_v1_00.tsv\amazon_reviews_us_Software_v1_00.tsv")
df = pd.read_csv(r"D:\GEC\Amazon reviews\amazon_reviews_us_Software_v1_00.tsv\amazon_reviews_us_Software_v1_00.tsv",  sep='\t', error_bad_lines=False)
clean_books = 
vocab_to_int = {}
count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# Add special tokens to vocab_to_int
codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1


letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ]


def noise_maker(sentence, threshold):
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0, 1, 1)
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0, 1, 1)
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    continue
                else:
                    noisy_sentence.append(sentence[i + 1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            else:
                pass
        i += 1
    return noisy_sentence

