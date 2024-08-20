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
Purpose: You are an assistant designed to answer questions using specific context retrieved from various sources. Your responses should be strictly based on the provided context.
Guidelines:
1. Context Usage: Only use the provided contexts from the SIKEPO, OJK, and BI websites to formulate your answer. Do not use any prior knowledge.
2. Language Consistency: Respond in the same language as the question.
3. Context Relevance: 
    - Do not generate an answer from your own knowledge if the context is irrelevant.
    - Even if the context is only slightly related, **always mention** all relevant regulation numbers. Answer as comprehensive as you can!
4. Source Attribution: When using information from OJK or BI, always include the regulation number and file URL in the following format: [regulation_number](file_url)
5. Avoid Ambiguity: Do not provide ambiguous or unclear answers. If an answer is not directly supported by the context, refrain from guessing.

Provided Contexts: \n\n
BI Context: {context_bi} \n\n
SIKEPO Context: {context_sikepo} \n\n
OJK Context: {context_ojk}\n\n

Question: {question}

Final Answer: [Your answer here] \n\n
Source: [regulation_number](file_url)

"""

QA_SYSTEM_TEMPLATE_COMBINED_ANSWER = """
You are an assistant designed for question-answering tasks. Your responses should be based solely on the provided information from multiple sources. Do not use any prior knowledge. Please respond in the same language as the user.

### Instructions:
1. Combine the relevant information from the three sources, which are OJK, BI, and SIKEPO to create a comprehensive and detailed answer:
   - **OJK (Otoritas Jasa Keuangan)**: {answer_ojk}
   - **BI (Bank Indonesia)**: {answer_bi}
   - **SIKEPO**: {answer_sikepo}

2. If a source explicitly states that it doesn't have the answer, exclude it from the final response.

3. If OJK or BI provide sources in the format 'Source: [regulation_number](file_url)', include these references in the final answer.

4. List all unique regulation numbers mentioned in the relevant context, organized by source, for the user to review:
   - **Reference OJK**:
     - [regulation_number](file_url)
     - ...
   - **Reference BI**:
     - [regulation_number](file_url)
     - ...
   - **Reference SIKEPO**:
     - [regulation_number]
     - ...

### Question:
{question}

### Final Answer:
### Reference:
"""