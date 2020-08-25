import textattack


if __name__ == '__main__':
    lm = textattack.constraints.grammaticality.language_models.Google1BillionWordsLanguageModel(5, 5)
    attack1 = textattack.shared.attacked_text.AttackedText("This is an example")
    original = textattack.shared.attacked_text.AttackedText("This is an sentence")
    print(lm.lm.get_words_probs("I am going to the", ["bank", "store", "monkey"]))

    glove = textattack.models.tokenizers.GloveTokenizer()
