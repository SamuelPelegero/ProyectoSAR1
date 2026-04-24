# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
#from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel



## UTILIZAR PARA LA AMPLIACION
# Selecciona un modelo semántico
SEMANTIC_MODEL = "SBERT"
#SEMANTIC_MODEL = "BetoCLS"
#SEMANTIC_MODEL = "Beto"
#SEMANTIC_MODEL = "Spacy"
#SEMANTIC_MODEL = "Spacy_noSW_noA"



def create_semantic_model(modelname):
    assert modelname in ("SBERT", "BetoCLS", "Beto", "Spacy", "Spacy_noSW_noA")
    
    if modelname == "SBERT": return SentenceBertEmbeddingModel()    
    elif modelname == "BetoCLS": return BetoEmbeddingCLSModel()
    elif modelname == "Beto": return BetoEmbeddingModel()
    elif modelname == "Spacy": SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
    return SpacyStaticModel()


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          posicionales + busqueda semántica + ranking semántico

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # campo que se indexa
    DEFAULT_FIELD = 'all'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10


    all_atribs = ['urls', 'index', 'docs', 'articles', 'tokenizer', 'show_all',
                  "semantic", "chuncks", "embeddings", "chunck_index", "kdtree", "artid_to_emb"]


    def __init__(self):
        """
        Constructor de la clase SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria pero
        	puedes añadir más variables si las necesitas. 

        """
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile(r"\W+") # expresion regular para hacer la tokenizacion
        self.show_all = False # valor por defecto, se cambia con self.set_showall()

        self.docid_counter = 0 # Para asignar IDs a los ficheros .json
        self.artid_counter = 0 # Para asignar IDs a cada artículo de la Wikipedia

        # PARA LA AMPLIACION
        self.semantic = None
        self.chuncks = []
        self.embeddings = []
        self.chunck_index = []
        self.artid_to_emb = {}
        self.kdtree = None
        self.semantic_threshold = None
        self.semantic_ranking = None # ¿¿ ranking de consultas binarias ??
        self.model = None
        self.MAX_EMBEDDINGS = 200 # número máximo de embedding que se extraen del kdtree en una consulta
        
        
        
        

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################


    def set_showall(self, v:bool):
        """

        Cambia el modo de mostrar los resultados.

        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v


    def set_semantic_threshold(self, v:float):
        """

        Cambia el umbral para la búsqueda semántica.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic es False el umbral no tendrá efecto.

        """
        self.semantic_threshold = v

    def set_semantic_ranking(self, v:bool):
        """

        Cambia el valor de semantic_ranking.

        input: "v" booleano.

        UTIL PARA LA AMPLIACIÓN

        si self.semantic_ranking es True se hará una consulta binaria y los resultados se rankearán por similitud semántica.

        """
        self.semantic_ranking = v


    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################


    def save_info(self, filename:str):
        """
        Guarda la información del índice en un fichero en formato binario

        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename:str):
        """
        Carga la información del índice desde un fichero en formato binario

        """
        #info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)


    ###############################
    ###                         ###
    ###   SIMILITUD SEMANTICA   ###
    ###                         ###
    ###############################

            
    def load_semantic_model(self, modelname:str=SEMANTIC_MODEL):
        """
    
        Carga el modelo de embeddings para la búsqueda semántica.
        Solo se debe cargar una vez
        
        """
        if self.model is None:
            print(f"loading {modelname} model ... ",end="", file=sys.stderr)             
            self.model = create_semantic_model(modelname)
            print("done!", file=sys.stderr)

            

    def update_chuncks(self, txt:str, artid:int):
        """
        
        Añade los chuncks (frases en nuestro caso) del texto "txt" correspondiente al articulo "artid" en la lista de chuncks
        Pasos:
            1 - extraer los chuncks de txt, en nuestro caso son las frases. Se debe utilizar "sent_tokenize" de la librería "nltk"
            2 - actualizar los atributos que consideres necesarios: self.chuncks, self.embeddings, self.chunck_index y self.artid_to_emb.
        
        """

        #1 - completar
        frases = nltk.sent_tokenize(txt)
        #2 - completar
        for f in frases:
            self.chunks.append(f)
            self.chunck_index.append(artid)

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
        self.model.fit(self.chunks)
        print("done!")


        
    def solve_semantic_query(self, query:str):
        """

        Resuelve una consulta utilizando el modelo semántico.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultados, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - si el último resultado tiene una distancia <= self.semantic_threshold 
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - también se puede salir si recuperamos todos los embeddings
            5 - tenemos una lista de chuncks que se debe pasar a artículos
        """

        self.load_semantic_model()
        
        # COMPLETAR

        top_k = self.MAX_EMBEDDINGS
        total_chunks = len(self.chuncks)
        #1,2
        while True:
            dists, idcs = self.model.query(query, top_k)
            
            # 3
            if self.semantic_threshold is not None and dists[-1] <= self.semantic_threshold:
                # 4
                if top_k < total_chunks:
                    top_k = min(top_k + self.MAX_EMBEDDINGS, total_chunks)
                    continue 
            break 

        # 5
        results = []
        for d, idc in zip(dists, idcs):
            if self.semantic_threshold is None or d <= self.semantic_threshold:
                artid = self.chunck_index[idc] 
                if artid not in results:
                    results.append(artid)
                    
        return results


    def semantic_reranking(self, query:str, articles: List[int]):
        """

        Ordena los articulos en la lista 'article' por similitud a la consulta 'query'.
        Pasos:
            1 - utiliza el método query del modelo sémantico
            2 - devuelve top_k resultado, inicialmente top_k puede ser MAX_EMBEDDINGS
            3 - a partir de los chuncks se deben obtener los artículos
            3 - si entre los artículos recuperados NO estan todos los obtenidos por la RI binaria
                  ==> no se han recuperado todos los resultado: vuelve a 2 aumentando top_k
            4 - se utiliza la lista ordenada del kdtree para ordenar la lista "articles"
        """
        
        self.load_semantic_model()
        top_k = self.MAX_EMBEDDINGS
        totalchunks = len(self.chuncks)
        artbus = set(articles)
        artord = []
        while True:
            dist, ids = self.model.query(query, top_k)
            encontrados = []
            for i in ids:
                aid = self.chunck_index[i]
                if aid in artbus and aid not in encontrados:
                    encontrados.append(aid)
            if len(encontrados) < len(articles) and top_k < totalchunks:
                top_k = min(top_k + self.MAX_EMBEDDINGS, totalchunks)
                continue

            artord = encontrados
            break

        for aid in articles:
            if aid not in artord:
                artord.append(aid)
                
        return artord
    

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article:Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls


    def index_dir(self, root:str, **args):
        """

        Recorre recursivamente el directorio o fichero "root"
        NECESARIO PARA TODAS LAS VERSIONES

        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.positional = args['positional']
        self.semantic = args['semantic']
        if self.semantic is True:
            self.load_semantic_model()


        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        #####################################################
        ## COMPLETAR SI ES NECESARIO FUNCIONALIDADES EXTRA ##
        #####################################################
        
        
    def parse_article(self, raw_line:str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """
        
        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections') # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article


    def index_file(self, filename:str):
        """
        Indexa el contenido de un fichero.
        Soporta indexación normal y posicional según self.positional.
        """
        # 1. Registrar el fichero en el diccionario de documentos
        current_docid = self.docid_counter
        self.docs[current_docid] = filename
        self.docid_counter += 1

        # 2. Abrir el fichero y leer línea a línea (usando utf-8 por seguridad)
        for i, line in enumerate(open(filename, encoding='utf-8')):
            
            j = self.parse_article(line)
            if j is None:
                continue

            # 3. Control de duplicados por URL
            url = j.get('url')
            if url in self.urls:
                continue
            self.urls.add(url)

            # 4. Asignar ID al artículo y guardar info de localización
            art_id = self.artid_counter
            self.articles[art_id] = {
                'docid': current_docid, 
                'line': i,              
                'title': j.get('title') 
            }
            self.artid_counter += 1

            # 5. Obtener el texto del campo por defecto
            texto = j.get(self.DEFAULT_FIELD, "")

            # 6. Tokenización
            terminos = self.tokenize(texto)

            # 7. Indexado en el índice invertido
            if getattr(self, 'positional', False):
                # --- VERSIÓN POSICIONAL ---
                # Estructura: {termino: [[art_id, [pos1, pos2]], [art_id2, [pos3]]]}
                for idx, term in enumerate(terminos):
                    if not term: continue
                    
                    if term not in self.index:
                        self.index[term] = []
                    
                    # Comprobamos si el último artículo añadido para este término es el actual
                    # Si la lista está vacía o el ID es distinto, creamos nueva entrada de artículo
                    if not self.index[term] or self.index[term][-1][0] != art_id:
                        self.index[term].append([art_id, [idx]])
                    else:
                        # Si ya estamos en el artículo actual, añadimos la posición a su lista
                        self.index[term][-1][1].append(idx)
            else:
                # --- VERSIÓN NO POSICIONAL (Mínima) ---
                # Estructura: {termino: [art_id1, art_id2, ...]}
                terminos_unicos = set(t for t in terminos if t) 

                for term in terminos_unicos:
                    if term not in self.index:
                        self.index[term] = []
                    
                    # Como procesamos artículos en orden, art_id siempre es mayor que el anterior
                    self.index[term].append(art_id)

    def tokenize(self, text:str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()




    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """

        print("-" * 30)
        print("ESTADÍSTICAS DEL ÍNDICE")
        print("-" * 30)
        print(f"Ficheros procesados: {len(self.docs)}")
        print(f"Artículos indexados: {self.artid_counter}")
        print(f"Palabras únicas: {len(self.index)}")
        
        # Vamos a imprimir solo 10 palabras para ver que el formato es correcto
        print("\nMuestra del índice (Primeras 10 palabras):")
        for i, (termino, posting) in enumerate(self.index.items()):
            if i < 10:
                print(f"  '{termino}': {posting}")
            else:
                break
        print("-" * 30)
        pass
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                             ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################


    def solve_query(self, query: str, prev: dict = {}):
        """
        Resuelve una query evaluando de izquierda a derecha.
        Soporta términos simples y el operador NOT.
        """
        if query is None or len(query.strip()) == 0:
            return [], None

        # 1. Tokenizar la consulta
        tokens = query.split()
        res = []
        i = 0

        # 2. Obtener la primera posting list (gestionando si empieza por NOT)
        try:
            if tokens[0].upper() == "NOT":
                # Caso: "NOT term1 ..." -> Invertimos la posting de term1
                if len(tokens) > 1:
                    res = self.reverse_posting(self.get_posting(tokens[1]))
                    i = 2
                else:
                    return [], None
            else:
                # Caso estándar: "term1 ..."
                res = self.get_posting(tokens[0])
                i = 1

            # 3. Procesar el resto de términos de izquierda a derecha
            while i < len(tokens):
                if tokens[i].upper() == "NOT":
                    # Si encontramos NOT, el siguiente término se resta (AND NOT)
                    if i + 1 < len(tokens):
                        next_p = self.get_posting(tokens[i + 1])
                        res = self.minus_posting(res, next_p)
                        i += 2
                    else:
                        i += 1 # NOT al final de la línea, lo ignoramos
                else:
                    # Si no hay NOT, es un AND implícito
                    next_p = self.get_posting(tokens[i])
                    res = self.and_posting(res, next_p)
                    i += 1

        except Exception as e:
            # En caso de error inesperado, devolvemos lista vacía para no romper el programa
            return [], None

        # Muy importante: devolver una tupla (resultado, metadata)
        return res, None

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################




    def get_posting(self, term:str):
        """

        Devuelve la posting list asociada a un termino.
        Puede llamar self.get_positionals: para las búsquedas posicionales.


        param:  "term": termino del que se debe recuperar la posting list.

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        term = term.lower()
        return self.index.get(term, [])
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    def get_positionals(self, terms:str):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LAS BÚSQUESAS POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.

        return: posting list

        """

        #################################
        ## COMPLETAR PARA POSICIONALES ##
        #################################
        pass



    def reverse_posting(self, p:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """
        
        notTotal = sorted(self.articles.keys())
        res = []
        i,j = 0,0
        while i < len(notTotal):
            if j < len(p) and notTotal[i] == p[j]:
                i += 1
                j += 1
            else:
                res.append(notTotal[i]) 
                i += 1
        return res
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################



    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        
        res = []
        i,j = 0,0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                res.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else: j += 1
        return res
        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################






    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        res = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                # Si son iguales, no lo añadimos (lo restamos) y avanzamos ambos
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                # Si el de p1 es menor, significa que no existe en p2 (porque p2 está ordenado)
                # Lo añadimos y avanzamos p1
                res.append(p1[i])
                i += 1
            else:
                # Si el de p1 es mayor, avanzamos p2 para intentar encontrarlo
                j += 1
        
        # Al terminar el bucle, si quedan elementos en p1, todos ellos pertenecen al resultado
        # porque ya no hay nada más en p2 que los pueda "restar".
        while i < len(p1):
            res.append(p1[i])
            i += 1
            
        return res
        
        ########################################################
        ## COMPLETAR PARA TODAS LAS VERSIONES SI ES NECESARIO ##
        ########################################################





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql:List[str], verbose:bool=True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r, _ = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results


    def solve_and_test(self, ql:List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result, _ = self.solve_query(query)
                result = len(result)
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)

        return not errors


    def solve_and_show(self, query: str):
        # 1. Resolver la query
        posts, _ = self.solve_query(query)
        num_results = len(posts)

        print(f"Query: '{query}'")
        print(f"Number of results: {num_results}")

        # 2. Determinar cuántos resultados mostrar
        res_to_show = posts if self.show_all else posts[:self.SHOW_MAX]

        # 3. Mostrar la información de cada artículo
        for i, art_id in enumerate(res_to_show, 1):
            info = self.articles[art_id]
            # Suponiendo que guardaste 'title' y 'url' en self.articles durante la indexación
            print(f"[{i}] ({art_id}) {info.get('title', 'No Title')}")
        
        print("-" * 20)
        ################
        ## COMPLETA  ##
        ################



