# Importer packages
from requests import get
import re
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

# Téléchargement des modules depuis nltk
nltk.download('punkt')
nltk.download('stopwords')

# fonction de nettoyage du code ...
def cleaner(text_to_clean):
    # Supprimer [\w]*
    text_to_clean = re.sub(r'[[\w]*]', ' ', text_to_clean)

    # Supprimer les chaines de \xa0, \u200c
    text_to_clean = re.sub(r'\xa0|\u200c', ' ', text_to_clean)

    # Remplacer les espaces multiples par l'espace simple
    text_to_clean = re.sub(r'/s+', ' ', text_to_clean)

    # Remplacer l'espace en debut et fin de corpus
    text_to_clean = re.sub(r'^\s|\s$', '', text_to_clean)
    
    return text_to_clean


# Fonction de calcul du poids des mots ...
def calculate_word_weight(text, language = 'english'):
    # Stopwords
    stopwords = nltk.corpus.stopwords.words(language)
    
    # Dictionnaire de fréquences des mots
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
                
    
    if(len(word_frequencies) > 0):
        maximum_frequency = max(word_frequencies.values()) # Fréquence maximale
        # Calculer la fréquence pondérée,
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / maximum_frequency

    return word_frequencies


# Fonction de calcul des scores des phrases ...
def get_sentences_score(text, language = 'english'):
    sentence_list = nltk.sent_tokenize(text)
    word_frequencies = calculate_word_weight(text, language = language)
    sentence_scores = {} # Liste des scores de chaque phrase
    
    # Calculer le score de chaque phrase
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
                        
    return sentence_scores


# Fonction de résume de texte ...
def resume_text(text, resume_size, language = 'english'):
    text = cleaner(text)
    sentence_scores = get_sentences_score(text, language = language)
    
    # Ordonner les phrases par pondération et recupérer les "resume_size" premières phrases
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=False)[:resume_size]
    
    # regrouper ensemble les phrases qui ont les poids les plus élévés
    summary = ' '.join(summary_sentences)

    # Afficher le résumé
    return summary


# Fonction de résume de texte  avec le modèle TextRankSummarizer...
def with_textRankSummarizer(text, resume_size, language = 'english'):
    # Importer le TextRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    # Initialiser le modèle
    summarizer_textrank = TextRankSummarizer()
    
    # cleanning text
    text = cleaner(text)
    
    # Créer un text parser utilisant de tokenisation
    parser = PlaintextParser.from_string(text, Tokenizer(language))

    # Summariser en 5 phrases
    summary = summarizer_textrank(parser.document, resume_size)

    # Regrouper les phrases
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)

    # Retourner le summary
    return text_summary


# Fonction de résume de texte  avec le modèle LexRankSummarizer...
def with_lexRankSummarizer(text, resume_size, language = 'english'):
    # Importer LexRankSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    
    # Initialiser le modèle
    summarizer_lexrank = LexRankSummarizer()
    
    # cleanning text
    text = cleaner(text)
    
    # Créer un text parser utilisant de tokenisation
    parser = PlaintextParser.from_string(text, Tokenizer(language))

    # Summariser en 5 phrases
    summary = summarizer_lexrank(parser.document, resume_size)

    # Regrouper les phrases
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)
        
    # Afficher le summary
    return text_summary


# Fonction de résume de texte  avec le modèle LsaSummarizer...
def with_lsaSummarizer(text, resume_size, language = 'english'):
    # Importer LsaSummarizer
    from sumy.summarizers.lsa import LsaSummarizer
    
    # Initialiser le modèle
    summarizer_lsa = LsaSummarizer()
    
    # cleanning text
    text = cleaner(text)
    
    # Créer un text parser utilisant de tokenisation
    parser = PlaintextParser.from_string(text, Tokenizer(language))

    # Summariser en 5 phrases
    summary = summarizer_lsa(parser.document, resume_size)

    # Regrouper les phrases
    text_summary = ""
    for sentence in summary:
        text_summary += str(sentence)

    # Afficher le summary
    return text_summary