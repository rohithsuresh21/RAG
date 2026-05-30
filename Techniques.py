#evaluation techniques

#1) ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
#2) BERTScore
#3) RAGAS (Retrieval-Augmented Generation Assessment Score)


# RoUGE is a set of metrics used to evaluate the quality of generated text by comparing it to reference texts. 
# It measures the overlap of n-grams, word sequences, and word pairs between the generated text and the reference texts. 
# ROUGE-N measures the overlap of n-grams, while ROUGE-L measures the longest common subsequence.
from rouge import Rouge
def evaluate_rouge(gen_txt, ref_txt):
    rouge = Rouge()
    scores = rouge.get_scores(gen_txt, ref_txt)
    return scores

# BERTScore is a metric that evaluates the similarity between generated text and reference text using contextual embeddings from BERT.
from bert_score import score
def evaluate_bertscore(gen_txt, ref_txt):
    P, R, F1 = score([gen_txt], [ref_txt], lang='en', verbose=False)
    return {'precision': P.item(), 'recall': R.item(), 'f1': F1.item()}

# RAGAS (Retrieval-Augmented Generation Assessment Score) is a metric designed to evaluate the quality of generated text in retrieval-augmented generation systems.
# It considers both the relevance of the retrieved information and the quality of the generated text.
def evaluate_ragas(gen_txt, ref_txt, retrieved_info):
    # This is a placeholder implementation. The actual RAGAS metric would require a more complex approach.
    relevance_score = compute_relevance(retrieved_info, ref_txt)
    generation_quality_score = compute_generation_quality(gen_txt, ref_txt)
    ragas_score = (relevance_score + generation_quality_score) / 2
    return ragas_score 

def compute_relevance(retrieved_info, ref_txt):
    # Placeholder for relevance computation
    return 0.8  # Example relevance score   

def compute_generation_quality(gen_txt, ref_txt):
    # Placeholder for generation quality computation
    return 0.7  # Example generation quality score

