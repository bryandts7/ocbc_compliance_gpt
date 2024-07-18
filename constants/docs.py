import ast
from  langchain.schema import Document

from database.database import rekam_jejak_docstore, ketentuan_terkait_docstore


def str_to_dict(s):
    # Replace the ObjectId string with a placeholder
    s = s.replace("ObjectId(", "").replace(")", "")

    # Use ast.literal_eval to safely evaluate the string
    d = ast.literal_eval(s)
    
    # Manually convert the _id field back to an ObjectId
    if '_id' in d:
        del d['_id']
    
    return d

def db_to_docs(docs):
    res = []
    for doc in docs:
        dic = str_to_dict(doc.page_content)
        lang_doc = Document(page_content=dic['page_content'], metadata=dic['metadata'])
        res.append(lang_doc)    
    return res

REKAM_DOCSTORE =  rekam_jejak_docstore().load()
KETENTUAN_DOCSTORE = ketentuan_terkait_docstore().load()

REKAM_DOCS = db_to_docs(REKAM_DOCSTORE)
KETENTUAN_DOCS = db_to_docs(KETENTUAN_DOCSTORE)