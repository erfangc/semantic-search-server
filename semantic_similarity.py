import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

text = (
    "Our operations primarily use Navisphere, our global, "
    "multimodal transportation management system that allows customers to communicate "
    "worldwide with parties in their supply chain across languages, currencies, and continents. "
    "Navisphere offers sophisticated business analytics, artificial intelligence, "
    "and data-driven tools to improve supply chain performance and meet increasing "
    "customer demands including the following tools"
)
doc = nlp(text)

comparisons = [sentence.__str__() for sentence in doc.sents]
query = ['use artificial intelligence to analyze customer behavior']

comparison_embeddings = model.encode(comparisons, convert_to_tensor=True)
query_embeddings = model.encode(query, convert_to_tensor=True)

# -----------------------
# |  cosine similarity
# -----------------------
cosine_scores = util.cos_sim(
    comparison_embeddings,
    query_embeddings,
)

for i in range(len(comparisons)):
    for j in range(len(query)):
        print("{} Score: {:.4f}".format(comparisons[i], cosine_scores[i][j]))

sum = 0
for i in range(len(comparisons)):
    sum = sum + cosine_scores[i][0]

print("Average score: {:.4f}".format(sum / len(comparisons)))
