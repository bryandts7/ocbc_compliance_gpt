from langchain_core.prompts import PromptTemplate

CONTEXTUALIZE_Q_PROMPT_STR = """Given the following conversation and a follow-up question, \
rephrase the follow-up question to be a standalone question in its original language. 
If the follow-up question is not clear, indicate so. 
If the chat history is not relevant to the follow-up question, please ignore the chat history.

Chat History:
{chat_history}

Follow-up Question: {question}
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
