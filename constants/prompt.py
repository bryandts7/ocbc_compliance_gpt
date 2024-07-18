ROUTER_PROMPT =  """You are an expert at routing a user question to the appropriate data source.
1.) User Inquiry about "Peraturan" or "Ketentuan" Status:
Criteria: If the user question asks about the relevance, modification, or history of "peraturan" or "ketentuan" (e.g., "Is this regulation still relevant?", "Has this rule been modified?", or any query related to "rekam jejak"),
Action: Return 'rekam_jejak'.

2.) User Inquiry for Detailed Explanation or Understanding:
Criteria: If the user question asks for detailed explanations, meanings of the regulations, or any queries unrelated to "rekam jejak" (e.g., "What does this regulation mean?", "Can you explain this rule in detail?", or any other unrelated questions),
Action: Return 'ketentuan_terkait'.
"""

RAG_FUSION_PROMPT = """You are a helpful assistant that generates 2 search queries based on a single input query. \n
    Generate 2 search queries related to: {question} \n

    If the question context is not really clear, you can utilize this text history based 
    on prior conversation between human and AI before this question being asked:
    {history}

    You just need to output ONLY QUERIES. DO NOT WRITE INDEXING OR ANY OTHER NOTES. The output template is like this:
    {{Original Query}}
    {{Paraphrased Query}}
    {{Paraphrased Query}}
    Output (1 ORIGINAL QUERY AND 2 PARAPHRASED QUERIES):"""

RAG_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Please do not use your prior knowledge."
    "If the question does not related to the context given, just answer 'Saya tidak tahu mengenai hal tersebut'. "
    "Use three (3) sentences maximum and keep the answer concise."
    "Please write your answer ONLY in INDONESIAN."

    # "Jika sebuah peraturan A mencabut atau mengubah peraturan B, C, D"
    # "maka peraturan B, C, D tersebut mungkin saja sudah tidak berlaku"
    
    # "Jika ada yang bertanya mengenai aturan masih berlaku atau masih relevan, "
    # "jawab 'Iya' ketika peraturan tersebut tidak diubah dan tidak dicabut peraturan lain dan jelaskan mengapa demikian."
    # "Sebaliknya, jawab 'Mungkin tidak berlaku' ketika ada peraturan lain yang mengubah atau mencabut peraturan tersebut. "
    # "Tambahkan peraturan apa yang mencabut atau mengubah jika jawaban nya 'Mungkin tidak berlaku' "
    "Jika pertanyaan menanyakan tentang peraturan, tulis dengan detail nomor ketentuan dan ketentuannya secara detail"
    "Jika konteks yang diberikan tidak relevan dengan pertanyaan, jangan pernah membuat jawaban dari pengetahuanmu sendiri"
    "Jawab kalau anda tidak tahu jika konteks yang diberikan tidak sesuai dengan pertanyaan."
    
    # "Please also mention the 'Nomor Ketentuan' and 'Ketentuan' from the metadata in a subtle way "
    # "such that you can add more context to answer the question. "
    "\n\n"
    "{context}"
)

RAG_REKAM_JEJAK_PROMPT = (
    "You will be provided with two sources of context: one from GraphRAG and one from RAG (which contains several Documents)."
    "Your task is to combine the results from these two contexts effectively. However, prioritize the context from GraphRAG if it is not empty."
    "If the context from GraphRAG is empty, then you should retrieve fully from the context provided by RAG."

    "If GraphRAG context provides information about a topic but is missing some details, supplement it with additional details from the RAG context."
    
    "GraphRAG Context:"
    "{structured}"
    "You can also rely on the Documents retreived here as an additional context:"
    "{unstructured}"
)

GRAPH_CYPHER_GEN_PROMPT = """Task:Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.
Jika seorang pengguna menanyakan apakah suatu peraturan masih relevan atau masih berlaku, Anda perlu memeriksa apakah peraturan tersebut telah "DICABUT" atau "DIUBAH" oleh peraturan lain, atau apakah peraturan lain "MENCABUT" atau "MENGUBAH" peraturan tersebut.

Example:
PLEASE USE THIS EXAMPLE FOR YOUR THINKING AND NEVER USE IT TO ANSWER ANY QUESTIONS.
User Query: Apakah peraturan dengan nomor XXXX masih berlaku?
Generated Cypher Query:

```cypher
MATCH (p:Peraturan {{nomor_ketentuan: 'XXXX'}})
OPTIONAL MATCH (p)<-[:MENCABUT|MENGUBAH]-(other:Peraturan)
OPTIONAL MATCH (p)-[:DICABUT|DIUBAH]->(newer:Peraturan)
RETURN 
  p AS originalRegulation, 
  other AS replacingOrRevokingRegulation, 
  newer AS replacedOrRevokedByRegulation,
  CASE 
    WHEN other IS NOT NULL OR newer IS NOT NULL THEN 'No, this regulation is no longer relevant.'
    ELSE 'Yes, this regulation is still relevant.'
  END AS relevanceStatus

The question is:
{question}

If the question is ambiguous like "Apakah peraturan TERSEBUT masih berlaku?", then you can use this prior conversation to retrieve what does it means by TERSEBUT:
{history}
"""

GRAPH_QA_GEN_PROMPT = """Anda adalah asisten yang mengambil hasil dari kueri Neo4j Cypher 
dan membentuk respons yang dapat dibaca manusia. Bagian hasil kueri berisi hasil kueri Cypher
yang dihasilkan berdasarkan pertanyaan bahasa alami pengguna. Informasi yang diberikan bersifat
otoritatif, Anda tidak boleh meragukannya atau mencoba menggunakan pengetahuan internal Anda untuk 
memperbaikinya. Jadikan jawabannya terdengar seperti respons terhadap pertanyaan.

Pertanyaan:
{question}

Jawaban dari pertanyaan di atas:
{context}


Jika informasi yang diberikan kosong, katakanlah Anda tidak tahu jawabannya.
Informasi kosong terlihat seperti ini: []

Jika informasinya tidak kosong, Anda harus memberikan jawaban menggunakan hasilnya. 
Informasi yang diberikan harusnya adalah jawaban dari pertanyaan yang ditanyakan.
Jika ada peraturan yang MENGUBAH, DIUBAH, MENCABUT, ATAU DICABUT
TOLONG JUGA masukkan informasi seperti nomor ketentuan dan informasi lainnya yang relevan sebagai bukti dari jawaban anda.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

WRITE YOUR ANSWER IN INDONESIAN LANGUAGE.
"""

SUMMARY_HISTORY_PROMPT = """Progressively summarize the lines of conversation provided and the previous summary returning a new summary with maximum length of  four (4) sentences for the new summary.
Please write in Indonesian Language only!

EXAMPLE
Current conversation:
-
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current conversation:
{summary}
{new_lines}

New summary:"""