import unittest
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.cna.cna_graph import CnaGraph
from rb.similarity import vector_model,vector_model_factory as stuff

class CnaTest(unittest.TestCase):

    def test_create_graph_en(self):
        text_string = "This is a text string. Does the parsing work?"
        doc = Document(Lang.EN, text_string)
        model_spec= [{"model":"word2vec","corpus":"coca"}]#,{"model":"lsa","corpus":"coca"}]

        models = [stuff.create_vector_model(Lang.EN,model=stuff.VectorModelType.from_str(model["model"]),corpus=model["corpus"]) for model in model_spec]
        models = [model for model in models if model is not None]

        
        # model = get_default_model(Lang.EN)
        cna = CnaGraph(doc,models)
        # self.assertEqual(len(cna.graph.edges), 4, "Should be 4")
        print(cna)
        
    
if __name__ == '__main__':
    unittest.main()
