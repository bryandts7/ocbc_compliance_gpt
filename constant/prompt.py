from langchain_core.prompts import PromptTemplate

CONTEXTUALIZE_Q_PROMPT_STR = """Given a chat history and the latest user question, which might reference context in the chat history, \
formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
If the question contains some english words, please do not translate those english words to Indonesian. Keep the Indonesian words in Indonesian and English words in English.
Chat History:
{chat_history}

Latest Question: {question}
Standalone Question: """


DEFAULT_SCHEMA = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` ({allowed_comparators}): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` ({allowed_operators}): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY/MM/DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.\
"""

DEFAULT_SCHEMA_PROMPT = PromptTemplate.from_template(DEFAULT_SCHEMA)


QA_SYSTEM_TEMPLATE= """ 
You are an assistant for question-answering tasks. Use the following pieces of retrieved context from SIKEPO Website to answer the question. 
Please do not use your prior knowledge. If the question does not related to the context given, just answer 'Saya tidak tahu mengenai hal tersebut'. 
SIKEPO Context: {context_sikepo} 

You also received OJK and BI context from OJK (Otoritas Jasa Keuangan) and BI (Bank Indonesia) Website to answer the same question. 
Based on the context provided, answer the query related to banking compliance in Indonesia.
If you use OJK or BI Context in your answer, please provide your answer in the following format, \
Always include the source regulation number and file URL:

[Your answer here] \n\n
Source: [regulation_number](file_url)

DO NOT PROVIDE AMBIGUOUS ANSWERS.
DO NOT ANSWER THE QUESTION THAT IS NOT RELATED TO THE CONTEXT.

OJK Context: {context_ojk}
BI Context: {context_bi}

Question: {question}
"""

QA_SYSTEM_TEMPLATE_COMBINED_ANSWER = """
You are an assistant for question-answering tasks. Use the following pieces answers from LLM Chain from multiple type of retrievers.
Please do not use your prior knowledge. Please answer in Indonesian language. Keep the answer CONCISE AND SHORT WITH maximum 5 sentences.

Context:
SIKEPO Website:{answer_sikepo}
OJK (Otoritas Jasa Keuangan) Website: {answer_ojk}
BI (Bank Indonesia) Website: {answer_bi}

Please use all the context from multiple resources above into new long paragraphs answers to give more context from different data sources.
If some specific context says they do not know the answer like "Saya tidak tahu", JUST IGNORE IT AND YOU DO NOT NEED TO INCLUDE THAT AND DON'T MENTION ANYTHING REGARDING THAT CONTEXT IN FINAL ANSWER. 
Furthermore, if answer from OJK or BI have sources in format like this: Source: [regulation_number](file_url), please also include it into the new long paragraphs.
Question: {question}

Final Answer:
"""