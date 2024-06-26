# Automatic Essay Grader

This repository contains the implementation of an automatic essay grader, a tool designed to assess and score essays written in natural language automatically. The automatic essay grader utilizes advanced natural language processing (NLP) techniques and machine learning algorithms to analyze various aspects of written essays, including semantics, coherence, grammar, and phrasology. The system aims to provide accurate and consistent evaluations of essays, facilitating efficient grading processes in educational and testing contexts.

## Challenges we faced
- Model Development: Developing accurate and reliable NLP models for essay grading posed a significant challenge. We encountered difficulties in training models that could effectively capture the nuances of human-written essays and provide consistent evaluations across different essay topics and writing styles.
-  Data Collection and Annotation: Acquiring a diverse and representative dataset of essays for training and testing purposes was --challenging. Additionally, annotating the dataset with accurate and reliable ground truth scores required significant effort and expertise.
- eature Engineering: Identifying and extracting relevant features from essays to train the grading models was a complex task. We experimented with various linguistic and structural features, such as word counts, sentence length, semantic similarity, and grammatical correctness, to improve the model's performance.
-   Explainability and Transparency: Ensuring transparency and interpretability in the grading process was a key concern. We faced challenges in designin


g explainability metrics and techniques that could provide clear insights into how the grading models arrive at their scores.
- Scalability and Efficiency: Developing a scalable and efficient grading system that could handle large volumes of essays in real-time was another challenge. We had to optimize the performance of the grading algorithms and implement parallel processing techniques to meet the system's requirements.

## Important Links
[Visit to Dataset](https://www.kaggle.com/c/asap-aes/overview)
[Visit to  GLOVE files] (https://drive.google.com/file/d/1Y6wISmPIAcM83aQBG2exkcrI9W7GsyFv/view?usp=sharing)
[visit to embedding pickle] (https://drive.google.com/file/d/1Y6wISmPIAcM83aQBG2exkcrI9W7GsyFv/view?usp=sharing)
- [x] Get pickle + GloVe files from Google Drive (or create your own) and place it into the DeepLearningFiles folder in mysite\grade

https://github.com/prajak002/SIT-ICOE-2024/assets/80170713/288f9322-4e2f-484f-ac72-3a70f409c3f2



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
```
# Hyperpaprameters for LSTM
Hidden_dim1=300
Hidden_dim2=100
return_sequences = True
dropout=0.5
recurrent_dropout=0.4
input_size=400
activation='relu'
bidirectional = True
batch_size = 64
epoch = 70
** hyperparameters for word2vec**
most_common_words= []
print(X.shape)
print(y.shape)
for traincv, testcv in cv_data:
    print("\n--------Fold {}--------\n".format(fold_count))
    # get the train and test from the dataset.
    X_train, X_test, y_train, y_test = X.iloc[traincv], X.iloc[testcv], y.iloc[traincv], y.iloc[testcv]
    train_essays = X_train['essay']
    #print("y_train",y_train)
    test_essays = X_test['essay']
    #y_train = torch.tensor(y_train,dtype=torch.long)
    train_sentences = []
    # print("train_essay ",train_essays.shape)
    #print(X_train.shape,y_train.shape)
    for essay in train_essays:
        # get all the sentences from the essay
        train_sentences.append(essay_to_wordlist(essay, remove_stopwords = True))

    # word2vec embedding
    print("Converting sentences to word2vec model")
    model,_ = build_word2vec(train_sentences, num_workers, num_features, min_word_count, context,
                  downsampling)
    top10 = collections.defaultdict(int)


```
## Usage

To use the automatic essay grader, follow the instructions provided in the documentation (`docs/`). Y
## Contributing

Contributors :- Aayush kumar Singh, Prajak Sen,Shubham kr. Singh ,Soumyonath Tripathy, Ujjwal Kumar 

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

