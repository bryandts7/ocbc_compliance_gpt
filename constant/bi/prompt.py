QA_SYSTEM_PROMPT_BI = """The context information is below.
Context: 
{context}

Based on the context and the metadata information provided, \
answer the query related to banking compliance in Indonesia.
Use the context and metadata information only, without relying on external sources.
ALWAYS ANSWER IN THE USER'S LANGUAGE.

Please provide your answer in the following format, \
including the regulation number and file URL if available:

[Your answer here] \n\n
Source: [metadata['title']](metadata['file_url'])

If you cannot find the regulation number, just provide the answer. 
If the file_url ends with '.pdf', you can add the metadata['page_number'] in the URL like this: 

[Your answer here] \n\n
Source: [metadata['title']](metadata['file_url#page=metadata['page_number']')

DO NOT PROVIDE AMBIGUOUS ANSWERS.
DO NOT ANSWER THE QUESTION THAT IS NOT RELATED TO THE CONTEXT.

Question: {question}
"""