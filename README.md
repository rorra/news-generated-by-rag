# Procesamiento del Lenguaje Natural - UP

Durante el proyecto, nos planteamos utilizar las tecnologías requeridas por las empresas hoy en día, relacionadas a NLP.
Basándonos en la experiencia laboral de Josué Menendez, quién es el único integrante del grupo trabajando con Inteligencia Artificial, nos proponemos aprender y trabajar sobre las siguientes tecnologías:
- RAG
- Qdrant (Vector database)
- Langchain
El proyecto va a consistir en la creación de generador de noticias mediante la técnica de RAG, el cuál recuperará información relevante de los sitios de noticias, y generará noticias para un sitio de noticias web. En otras palabras, un sitio de noticias generado por inteligencia artificial.
El proyecto consistirá desde el web scraping de sitios de noticias, almacenamiento de los datos en una base de datos vectorial, elección del modelo LLM a utilizar, generación de noticias con refrescos de cada 1 hora en un sitio web, y deploy en entorno productivo en AWS.

# Esquema propuesto para el plan del proyecto

## Planteamiento del problema

Generación de un sitio de noticias por medio de la inteligencia artificial a través de la técnica RAG.
El proyecto está acotado a noticias para el público Argentino, con el lenguaje Español.

## Tecnologías a utilizar

- RAG
- Qdrant
- Modelo: GPT, Llama, Claude… a decidir
- Langchain
- Crewai ? https://github.com/crewAIInc/crewAI

## Pasos

1. **Web Scraping**: Obtención de las últimas noticias de sitios de noticias mediante web scraping, APIs.
   - Limitar el scraping a 4 secciones.
   - **Fecha límite: Septiembre 23**

2. **Data Pipeline**: Pre procesamiento de los datos, para darle una forma y quedarnos con lo que nos interesa.

3. **Preprocesamiento y Vectorización**: Embeddings, vectorizar, construir la base de datos vectorial.

4. **Módulo de Recuperación (Retrieval)**: API de búsqueda de la base de datos vectorial. Este módulo recupera noticias relevantes desde la base de datos vectorial basándose en consultas específicas (como temas, palabras clave, o incluso preguntas de los usuarios). Se pueden implementar consultas predefinidas para temáticas comunes.
   - **Fecha límite: Septiembre 29**

5. **Módulo de Generación (Generation)**: Este módulo utiliza un modelo de lenguaje para generar nuevos textos. Estos textos pueden ser nuevas noticias, resúmenes, o artículos de opinión basados en la información recuperada. Probar modelos LLM, prompt engineering, quedarnos con el que menos recursos consuma y cumpla con las expectativas.
   - Datos estáticos acotados para poder realizar las evaluaciones.
   - Evaluar **Content Retrieval**
     - MRR / MAP (?)
   - Evaluar **Generación**
   - **Fecha límite: Octubre 5**

6. **Almacenamiento y Publicación de Contenidos Generados**: Los textos generados se almacenan en una base de datos, y se construye una interfaz web para visualizar los mismos. La interfaz se construye en formato HTML/CSS, para una rápida visualización de las noticias.
   - **Fecha límite: Octubre 11**

7. **Monitorización y Mejora Continua**: Monitorización de los diferentes procesos para asegurar que el sitio esté disponible 24x7.
   - **Fecha límite: Octubre 25**

### Opcional
- AI Agent para publicar links de las noticias en X.com.
- AI Agent para publicar noticias con las fotos en Instagram.
- Generación de noticias en otros idiomas.


# Sitios de noticias a parsear

### Página 12
- https://www.pagina12.com.ar/
- Economía: https://www.pagina12.com.ar/secciones/economia
- Internacional: https://www.pagina12.com.ar/secciones/el-mundo
- Sociedad: https://www.pagina12.com.ar/secciones/sociedad

### TN
- https://tn.com.ar/
- Economía: https://tn.com.ar/economia/
- Internacional: https://tn.com.ar/internacional/
- Política: https://tn.com.ar/politica/
- Sociedad: https://tn.com.ar/sociedad/

### Perfil
- https://www.perfil.com/
- Economía: https://www.perfil.com/seccion/economia
- Internacional: https://www.perfil.com/seccion/internacional
- Política: https://www.perfil.com/seccion/politica
- Sociedad: https://www.perfil.com/seccion/sociedad

### Infobae
- https://www.infobae.com/?noredirect
- Economía: https://www.infobae.com/economia/
- Internacional: https://www.infobae.com/america/
- Política: https://www.infobae.com/politica/
- Sociedad: https://www.infobae.com/cultura/

### El economista
- https://eleconomista.com.ar/
- Economía: https://eleconomista.com.ar/economia/
- Internacional: https://eleconomista.com.ar/internacional/
- Politica: https://eleconomista.com.ar/politica/

### Ámbito Financiero
- https://www.ambito.com/
- Economía: https://www.ambito.com/contenidos/economia.html
- Politica: https://www.ambito.com/contenidos/politica.html

