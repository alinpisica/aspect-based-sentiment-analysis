import os
import xml.etree.ElementTree as ET
from data_models import Opinion, Sentence, Review
from utils import decontract_phrase

class SemevalReader():
    def __init__(self, xmlpath):
        self.xmlpath = xmlpath
        self.reviews = []
    
    def read_reviews(self):
        xml_data = open(os.path.join(self.xmlpath), 'r').read()
        root = ET.XML(xml_data)

        reviews = []

        for _, review in enumerate(root):
            current_sentences = []

            for _, sentences in enumerate(review):
                for _, sentence in enumerate(sentences):
                    opinions = []
                    text = ""

                    for _, sentence_children in enumerate(sentence):
                        if sentence_children.tag == 'Opinions':
                            for _, opinion in enumerate(sentence_children):
                                op = Opinion(
                                    target = opinion.attrib["target"] if "target" in opinion.attrib else "",
                                    category = opinion.attrib["category"] if "category" in opinion.attrib else "",
                                    polarity = opinion.attrib["polarity"] if "polarity" in opinion.attrib else "",
                                    start = opinion.attrib["from"] if "from" in opinion.attrib else "",
                                    end = opinion.attrib["to"] if "to" in opinion.attrib else ""
                                )
                                opinions.append(op)
                        if sentence_children.tag == "text":
                            text = decontract_phrase(sentence_children.text)

                    sen = Sentence(text, opinions)
                    current_sentences.append(sen)
                    
            newReview = Review(current_sentences)
            reviews.append(newReview)
        
        self.reviews = reviews

        return reviews
    
    def get_absolute_polarity_sentences(self):
        absolute_polarity_sentences = []

        for review in self.reviews:
            for sentence in review.sentences:
                if len(set(map(lambda x: x.polarity, sentence.opinions))) == 1:
                    absolute_polarity_sentences.append(sentence)
        
        return absolute_polarity_sentences
    
    def get_target_list_for_polarity(polarity):
        if polarity == 'positive':
            return [0, 0, 1]
        if polarity == 'negative':
            return [1, 0, 0]
        return [0, 1, 0]

    def get_polarity_from_target_list(target_list):
        if target_list == [0, 0, 1]:
            return 'positive'
        if target_list == [1, 0, 0]:
            return 'negative'
        return 'neutral'