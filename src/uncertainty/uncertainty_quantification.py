import numpy as np
from sentence_transformers import SentenceTransformer

def calculate_uncertainty(sentences: list[str]) -> float:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentences, convert_to_tensor=False)
    T = model.similarity(embeddings, embeddings).numpy()  
    t_mean = np.mean(T, axis=1, keepdims=True) 
    n = T.shape[1]
    if n <= 1:
        return 0.0     
    uncertainty = np.sum(np.linalg.norm(T - t_mean, axis=1)**2) / (n - 1)
    return float(uncertainty)

no_uncertainty = [
        "The capital of France is Paris.",
        "The capital of France is Paris.",
        "The capital of France is Paris.",
]
print(f"No uncertainty: {calculate_uncertainty(no_uncertainty)}")

low_uncertainty = [
        "The dog chased the cat up the tree.",
        "A hound pursued a feline up a tree.",
        "The puppy ran after the cat into the branches.",
]
print(f"Low uncertainty: {calculate_uncertainty(low_uncertainty)}")

high_uncertainty = [
        "Python is a great programming language.",
        "Bananas are yellow fruits grown in tropical climates.",
        "The standard speed of light is roughly 300,000 kilometers per second.",
]

print(f"High uncertainty: {calculate_uncertainty(high_uncertainty)}")


