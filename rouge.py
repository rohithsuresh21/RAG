from rouge_score import rouge_scorer

reference = "the system retrieved the correct context."
generated = "the system retrieved context."

#ROUGE-1 (unigram overlap) and ROUGE-2 (bigram overlap)
score = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
scores = score.score(reference, generated)

r1= scores['rouge1']
r2= scores['rouge2']
print(f"ROUGE-1: Precision: {r1.precision:.4f}, Recall: {r1.recall:.4f}, F1: {r1.fmeasure:.4f}")
print(f"ROUGE-2: Precision: {r2.precision:.4f}, Recall: {r2.recall:.4f}, F1: {r2.fmeasure:.4f}")
