# versión 1.2

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle
import nltk
from SAR_semantics import SentenceBertEmbeddingModel, BetoEmbeddingCLSModel, BetoEmbeddingModel, SpacyStaticModel



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
    elif modelname == "Spacy": return SpacyStaticModel(remove_stopwords=False, remove_noalpha=False)
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
        self.positional = False
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
        
        if self.kdtree is not None:
            self.load_semantic_model()
            self.model.kdtree = self.kdtree


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
        inicio = len(self.chuncks)
        for f in frases:
            self.chuncks.append(f)
            self.chunck_index.append(artid)
        
        self.artid_to_emb[artid] = (inicio, len(self.chuncks))

    def create_kdtree(self):
        """
        
        Crea el tktree utilizando un objeto de la librería SAR_semantics
        Solo se debe crear una vez despues de indexar todos los documentos
        
        # 1: Se debe llamar al método fit del modelo semántico
        # 2: Opcionalmente se puede guardar información del modelo semántico (kdtree y/o embeddings) en el SAR_Indexer
        
        """
        print(f"Creating kdtree ...", end="")
        self.model.fit(self.chuncks)
        self.kdtree = self.model.kdtree
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
        top_k = self.MAX_EMBEDDINGS
        total_chunks = len(self.chuncks)

        while True:
            top_k = min(top_k, total_chunks)
            # CAMBIO: Recibimos una sola variable 'res' (lista de tuplas)
            res = self.model.query(query, top_k)
            
            if not res: break

            # Accedemos a la última tupla para la condición de salida
            last_dist = res[-1][0] # res[-1] es (distancia, indice)

            if (self.semantic_threshold is not None
                    and last_dist <= self.semantic_threshold
                    and top_k < total_chunks):
                top_k = min(top_k + self.MAX_EMBEDDINGS, total_chunks)
                continue
            break

        results = []
        # CAMBIO: Iteramos sobre las tuplas (distancia, indice)
        for d, idc in res:
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
        total_chunks = len(self.chuncks)
        art_set = set(articles)
        
        while True:
            top_k = min(top_k, total_chunks)
            # CAMBIO: Aquí estaba el ValueError. Recibe solo 'res'.
            res = self.model.query(query, top_k)
            
            # Extraemos los art_id únicos que están en la lista original
            reranked = []
            for _, idc in res: # Solo nos interesa el índice (idc)
                aid = self.chunck_index[idc]
                if aid in art_set and aid not in reranked:
                    reranked.append(aid)
            
            # Si ya encontramos todos los artículos o no hay más chunks, paramos
            if len(reranked) >= len(articles) or top_k >= total_chunks:
                break
                
            top_k = min(top_k + self.MAX_EMBEDDINGS, total_chunks)

        # Añadir al final los que falten (por seguridad)
        for aid in articles:
            if aid not in reranked:
                reranked.append(aid)

        return reranked
    

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

        if self.semantic:
            self.create_kdtree()


        
        
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
                'title': j.get('title'),
                'url': j.get('url') 
            }
            self.artid_counter += 1

            # 5. Obtener el texto del campo por defecto
            texto = j.get(self.DEFAULT_FIELD, "")

            # 6. Tokenización
            terminos = self.tokenize(texto)

            # 7. Indexado en el índice invertido
            if self.positional:
                # --- VERSIÓN POSICIONAL ---
                # Estructura: {termino: [[art_id, [pos1, pos2, ...]], [art_id2, [pos3, ...]], ...]}
                for idx, term in enumerate(terminos):
                    if not term:
                        continue
                    if term not in self.index:
                        self.index[term] = []
                    # Si el último artículo añadido para este término ya es el actual, añadimos la posición
                    if self.index[term] and self.index[term][-1][0] == art_id:
                        self.index[term][-1][1].append(idx)
                    else:
                        # Nuevo artículo para este término
                        self.index[term].append([art_id, [idx]])
            else:
                # --- VERSIÓN NO POSICIONAL (Mínima) ---
                # Estructura: {termino: [art_id1, art_id2, ...]}
                terminos_unicos = set(t for t in terminos if t)
                for term in terminos_unicos:
                    if term not in self.index:
                        self.index[term] = []
                    # art_id es siempre mayor que el anterior (orden secuencial)
                    self.index[term].append(art_id) 

            if self.semantic:
                self.update_chuncks(texto, art_id)

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
        print(f"  Ficheros (docs) procesados : {len(self.docs)}")
        print(f"  Artículos indexados        : {self.artid_counter}")
        print(f"  Términos únicos en índice  : {len(self.index)}")
        print(f"  Índice posicional          : {self.positional}")
        print(f"  Índice semántico           : {bool(self.semantic)}")
        if self.semantic:
            print(f"  Chunks semánticos          : {len(self.chuncks)}")
        print("=" * 40)
       
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
        if not query:
            return [], None

        if self.semantic_threshold is not None and not self.semantic_ranking:
            results = self.solve_semantic_query(query)
            return results, None
 
        # --- PARSEO DE LA QUERY ---
        # Dividimos la query en tokens respetando las frases entre comillas
        # Ejemplo: 'python "fin de semana" NOT curso' ->
        #   tokens = ['python', '"fin de semana"', 'NOT', 'curso']
        tokens = self._parse_query_tokens(query)
 
        if not tokens:
            return [], None
 
        res = []
        i = 0
 
        # --- PRIMER TOKEN ---
        if tokens[0].upper() == "NOT":
            # NOT al inicio: complemento de la posting del siguiente término/frase
            next_posting = self._get_posting_for_token(tokens[1])
            res = self.reverse_posting(next_posting)
            i = 2
        else:
            res = self._get_posting_for_token(tokens[0])
            i = 1
 
        # --- RESTO DE TOKENS ---
        while i < len(tokens):
            token_upper = tokens[i].upper()
 
            if token_upper == "NOT":
                # AND NOT: intersección de res con el complemento del siguiente término
                # FIX: usamos and_posting(res, reverse_posting(next_p)) en lugar de minus_posting
                # para obtener el complemento global correcto (artículos que NO contienen el término)
                next_posting = self._get_posting_for_token(tokens[i + 1])
                res = self.and_posting(res, self.reverse_posting(next_posting))
                i += 2
            else:
                # AND implícito: intersección de res con la posting del término actual
                next_posting = self._get_posting_for_token(tokens[i])
                res = self.and_posting(res, next_posting)
                i += 1
 
        # Normalizar a lista de IDs puros (sin posiciones)
        final_res = [x[0] if isinstance(x, list) else x for x in res]
 
        # --- RERANKING SEMÁNTICO (opción -R) ---
        if self.semantic_ranking and self.semantic:
            final_res = self.semantic_reranking(query, final_res)
 
        return final_res, None
 
 
    def _parse_query_tokens(self, query:str) -> List[str]:
        """
        Método auxiliar: divide la query en tokens respetando las frases entre comillas dobles.
 
        Ejemplo:
            'python "fin de semana" NOT curso'
            -> ['python', '"fin de semana"', 'NOT', 'curso']
 
        Las frases entre comillas se devuelven con las comillas incluidas para que
        _get_posting_for_token pueda identificarlas.
        """
        tokens = []
        # Expresión regular: captura texto entre comillas O palabras sueltas
        pattern = re.compile(r'"[^"]*"|\S+')
        for match in pattern.finditer(query):
            tokens.append(match.group())
        return tokens
    
    def _get_posting_for_token(self, token:str) -> list:
        """
        Método auxiliar: devuelve la posting list para un token, que puede ser:
          - Una frase entre comillas (búsqueda posicional): llama a get_positionals
          - Un término simple: llama a get_posting
        """
        if token.startswith('"') and token.endswith('"'):
            # Búsqueda posicional: extraemos el contenido entre comillas
            frase = token[1:-1]
            return self.get_positionals(frase)
        else:
            return self.get_posting(token)




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

        # 1. Tokenizamos la frase
        tokens = self.tokenize(terms)
        if not tokens:
            return []
 
        # 2. Obtenemos las postings completas (con posiciones)
        postings_con_pos = [self.index.get(t, []) for t in tokens]
 
        # Si alguna palabra no existe en el índice, la frase no puede aparecer
        if any(not p for p in postings_con_pos):
            return []
 
        # 3. Intersección de IDs: artículos que contienen todas las palabras
        # Extraemos sólo los IDs para el and_posting
        ids_0 = [x[0] for x in postings_con_pos[0]]
        common_artids = ids_0
        for p in postings_con_pos[1:]:
            ids_p = [x[0] for x in p]
            common_artids = self.and_posting(common_artids, ids_p)
 
        res = []
        # 4. Comprobación de consecutividad para cada artículo común
        for aid in common_artids:
            # Extraemos las listas de posiciones de cada palabra para este artículo
            pos_por_palabra = []
            for p_list in postings_con_pos:
                for entry in p_list:
                    if entry[0] == aid:
                        pos_por_palabra.append(set(entry[1]))  # set para O(1) en búsqueda
                        break
 
            # Comprobamos si existe alguna posición inicial desde la que la secuencia es consecutiva
            for start_pos in pos_por_palabra[0]:
                es_frase = True
                for offset in range(1, len(pos_por_palabra)):
                    if (start_pos + offset) not in pos_por_palabra[offset]:
                        es_frase = False
                        break
                if es_frase:
                    res.append(aid)
                    break  # Basta con encontrar la frase una vez en el artículo
 
        return res



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
        i, j = 0, 0
        
        while i < len(notTotal):
            # Extraemos el ID de p si es posicional
            val_p = -1 # Valor por defecto si j está fuera de rango
            if j < len(p):
                val_p = p[j][0] if self.positional else p[j]

            if j < len(p) and notTotal[i] == val_p:
                # Si el artículo total está en la lista p, lo ignoramos (NOT)
                i += 1
                j += 1
            else:
                # Si no está en p, lo añadimos al resultado
                # Nota: El resultado de un NOT siempre es una lista de IDs (sin posiciones)
                # porque los artículos que NO tenían la palabra no tienen posiciones que guardar
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
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            # Si es posicional, comparamos p1[i][0], si no, p1[i]
            val1 = p1[i][0] if isinstance(p1[i], list) else p1[i]
            val2 = p2[j][0] if isinstance(p2[j], list) else p2[j]

            if val1 == val2:
                res.append(val1) # Guardamos solo el ID para el resultado booleano
                i += 1
                j += 1
            elif val1 < val2:
                i += 1
            else:
                j += 1
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
            # Extraemos los IDs según el tipo de índice
            val1 = p1[i][0] if isinstance(p1[i], list) else p1[i]
            val2 = p2[j][0] if isinstance(p2[j], list) else p2[j]

            if val1 == val2:
                # Si está en ambos, se resta: no se añade y avanzamos ambos
                i += 1
                j += 1
            elif val1 < val2:
                # Si val1 es menor, no puede estar en p2 (están ordenadas)
                # IMPORTANTE: Añadimos el elemento original (con posiciones si las hay)
                res.append(p1[i])
                i += 1
            else:
                # val1 es mayor, avanzamos p2 para buscarlo más adelante
                j += 1
        
        # Añadimos el resto de elementos de p1 que no han sido restados
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
            title = info.get('title', 'Sin título')
            url = info.get('url', 'Sin URL')
            # Formato: [orden] (artid) título \t url
            print(f"[{i}]\t({art_id})\t{title}\t{url}")
 
        print("-" * 40)



