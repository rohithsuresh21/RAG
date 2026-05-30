# 1. various evaluation techniques for RAG

# 1) BLEU (Bilingual Evaluation Understudy): This metric evaluates the quality of generated text by comparing it to reference texts. It measures the n-gram overlap between the generated text and the reference texts, providing a score that indicates how closely the generated text matches the reference.
from nltk.translate.bleu_score import sentence_bleu