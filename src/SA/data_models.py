class Opinion:
    def __init__(self, target, category, polarity, start, end):
        self.target = target
        self.category = category
        self.polarity = polarity
        self.start = start
        self.end = end

class Sentence:
    def __init__(self, text, opinions):
        self.text = text
        self.opinions = opinions

class Review:
    def __init__(self, sentences):
        self.sentences = sentences