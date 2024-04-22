# Automatic Essay Grader

This repository contains the implementation of an automatic essay grader, a tool designed to assess and score essays written in natural language automatically. The automatic essay grader utilizes advanced natural language processing (NLP) techniques and machine learning algorithms to analyze various aspects of written essays, including semantics, coherence, grammar, and phrasology. The system aims to provide accurate and consistent evaluations of essays, facilitating efficient grading processes in educational and testing contexts.

## Challenges we faced
- Model Development: Developing accurate and reliable NLP models for essay grading posed a significant challenge. We encountered difficulties in training models that could effectively capture the nuances of human-written essays and provide consistent evaluations across different essay topics and writing styles.
-  Data Collection and Annotation: Acquiring a diverse and representative dataset of essays for training and testing purposes was --challenging. Additionally, annotating the dataset with accurate and reliable ground truth scores required significant effort and expertise.
- eature Engineering: Identifying and extracting relevant features from essays to train the grading models was a complex task. We experimented with various linguistic and structural features, such as word counts, sentence length, semantic similarity, and grammatical correctness, to improve the model's performance.
-   Explainability and Transparency: Ensuring transparency and interpretability in the grading process was a key concern. We faced challenges in designing explainability metrics and techniques that could provide clear insights into how the grading models arrive at their scores.
- Scalability and Efficiency: Developing a scalable and efficient grading system that could handle large volumes of essays in real-time was another challenge. We had to optimize the performance of the grading algorithms and implement parallel processing techniques to meet the system's requirements.

## Important Links
[Visit to Dataset](https://www.kaggle.com/c/asap-aes/overview)
[Visit to  GLOVE files] (https://drive.google.com/file/d/1Y6wISmPIAcM83aQBG2exkcrI9W7GsyFv/view?usp=sharing)
[visit to embedding pickle] (https://drive.google.com/file/d/1Y6wISmPIAcM83aQBG2exkcrI9W7GsyFv/view?usp=sharing)
[x] Get pickle + GloVe files from Google Drive (or create your own) and place it into the DeepLearningFiles folder in mysite\grade



## HOW TO SET UP WEB APP

```
pip install django 3
pip install tensorflow
pip install keras
pip install gensim
pip install pyspellchecker
pip install --upgrade language_tool_python
pip install nltk
nltk.download(necesssary corpora and models(punkt, stopwords, wordnet)
pip install django-extensions

```
## How to run
```
python manage.py migrate
python manage.py runserver
```
## Usage

To use the automatic essay grader, follow the instructions provided in the documentation (`docs/`). Y
## Contributing

Contributors :- Aayush kumar Singh, Prajak Sen,Shubham kr. Singh ,Soumyonath Tripathy, Ujjwal Kumar 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

