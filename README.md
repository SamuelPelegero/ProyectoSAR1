# 🔎 Wikipedia Search Engine

Motor de búsqueda en Python para artículos de Wikipedia en español. Implementa un **índice invertido** con búsquedas booleanas y de frases exactas, y una ampliación de **búsqueda semántica** basada en embeddings de frases.

Proyecto desarrollado para la asignatura SAR (Sistemas de Almacenamiento y Recuperación de Información).

## ✨ Características

- **Indexación** de colecciones de artículos en formato JSON, con control de duplicados por URL.
- **Índice invertido** con posting lists ordenadas.
- **Índice posicional** (`-P`) para consultas de frases exactas entre comillas.
- **Consultas booleanas** con `AND` implícito y operador `NOT`.
- **Búsqueda semántica** (`-S`): recupera artículos cuyas frases sean semánticamente cercanas a la consulta, usando un umbral de distancia.
- **Ranking semántico** (`-R`): ordena los resultados de una consulta booleana por similitud semántica.
- Varios **modelos de embeddings** intercambiables: SBERT, BETO (media y CLS) y vectores estáticos de spaCy.
- Guardado y carga del índice en disco (`pickle`).
- Modos de ejecución: consulta única, lista de consultas, test contra resultados de referencia e interactivo.

## 🛠️ Tecnologías

- Python 3
- NLTK (segmentación en frases)
- Sentence-Transformers, Transformers y PyTorch (embeddings)
- spaCy (`es_core_news_lg`)
- scikit-learn (`KDTree` para la búsqueda de vecinos más cercanos)
- NumPy

## 📁 Estructura del proyecto

| Fichero | Descripción |
|---|---|
| `SAR_lib.py` | Clase `SAR_Indexer`: indexación, recuperación, búsqueda semántica y presentación de resultados. |
| `SAR_semantics.py` | Modelos de embeddings (`SentenceBertEmbeddingModel`, `BetoEmbeddingModel`, `BetoEmbeddingCLSModel`, `SpacyStaticModel`) y construcción del KDTree. |
| `SAR_indexer.py` | Script de línea de comandos para crear el índice. |
| `SAR_searcher.py` | Script de línea de comandos para realizar consultas. |

> Ajusta los nombres de los scripts si en tu repositorio se llaman de otra forma.

## ⚙️ Instalación

```bash
git clone https://github.com/SamuelPelegero/ProyectoSAR1.git
cd ProyectoSAR1

pip install numpy scikit-learn nltk spacy torch transformers sentence-transformers
python -m spacy download es_core_news_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## 🚀 Uso

### 1. Crear el índice

```bash
# Índice básico
python SAR_indexer.py <directorio_articulos> <nombre_indice>

# Índice posicional
python SAR_indexer.py <directorio_articulos> <nombre_indice> -P

# Índice posicional + semántico
python SAR_indexer.py <directorio_articulos> <nombre_indice> -P -S
```

Al terminar se muestran estadísticas (ficheros, artículos, términos únicos, tiempo de indexación y de guardado).

### 2. Realizar consultas

```bash
# Consulta única
python SAR_searcher.py <nombre_indice> -Q "python NOT curso"

# Frase exacta (requiere índice posicional)
python SAR_searcher.py <nombre_indice> -Q '"fin de semana"'

# Solo el número de resultados
python SAR_searcher.py <nombre_indice> -C -Q "universidad valencia"

# Mostrar todos los resultados (por defecto, 10)
python SAR_searcher.py <nombre_indice> -A -Q "universidad valencia"

# Lista de consultas / test con resultados de referencia
python SAR_searcher.py <nombre_indice> -L consultas.txt
python SAR_searcher.py <nombre_indice> -T consultas_test.txt

# Modo interactivo (sin -Q, -L ni -T)
python SAR_searcher.py <nombre_indice>
```

### 3. Búsqueda semántica

El índice debe haberse creado con `-S`.

```bash
# Búsqueda semántica con umbral de distancia
python SAR_searcher.py <nombre_indice> -S 0.8 -Q "animales que viven en el mar"

# Consulta booleana con ranking semántico
python SAR_searcher.py <nombre_indice> -R -Q "cine español"
```

`-S` y `-R` son mutuamente excluyentes. El modelo de embeddings se elige en la constante `SEMANTIC_MODEL` de `SAR_lib.py` (`SBERT` por defecto).

## 🧠 Cómo funciona

1. **Indexación:** cada artículo se tokeniza (minúsculas y eliminación de símbolos no alfanuméricos) y sus términos se añaden al índice invertido. En modo posicional, cada posting guarda también las posiciones del término.
2. **Consultas booleanas:** las posting lists se combinan con algoritmos de *merge* lineales sobre listas ordenadas (`AND`, `NOT`). Las frases entre comillas se resuelven comprobando posiciones consecutivas.
3. **Semántica:** el texto de cada artículo se divide en frases, se calcula un embedding por frase y se construye un `KDTree`. Al consultar, se recuperan las frases más cercanas (ampliando el número de vecinos hasta cubrir el umbral) y se traducen a artículos.
4. **Ranking semántico:** los artículos devueltos por la consulta booleana se reordenan según la posición de sus frases en la lista de vecinos más cercanos.

