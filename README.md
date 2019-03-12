**NLP Projects**

1.  **Toxic Comments Classification** - Build a multi-headed model that predicts
    probabilities of different types of toxicity such as threats, obscenity,
    insults, and identity-based in the comments.

    1.  **Metrics**: AUC ROC

    2.  **Models used:** Logistic Regression and RNN (LSTM/GRU)

    3.  **Approach:**

        1.  Used **Matplotlib** and **Searborn** to visualize distribution of
            target classes, correlation of categorical variables using
            **confusion matrix** and frequently used words using **WordCloud**.

        2.  Created more features and visualized those features to understand
            whether longer comments are more toxic, etc

        3.  Vectorized the words using **TF-IDF** and used
            **LogisticRegression** with additional features created from step 2
            and without those features.

        4.  Corrected spelling mistakes using **FastText embeddings** and
            trained a **RNN** network with **LSTM** and **GRU** cells using
            **Keras**.

2.  **StackOverFlow Tag Prediction** - Suggest tags based on the title and
    question text (multi-class classification).

    1.  **Metrics**: Mean F1 Score

    2.  **Models used**: Logistic Regression with OneVsRest Classifier,
        SGDClassifier with hinge loss

    3.  **Approach**:

        1.  Used Matplotlib and Searborn to visualize distribution of tags, tags
            per question and most frequent tags.

        2.  Preprocessed the data to remove special characters, HTML tags and
            **SnowballStemmer** to stem the words. Saved the cleaned data to
            **SQLite** DB.

        3.  Vectorized the words using **TF-IDF** and used **LogisticRegression
            with One Vs Rest Classifier** and **SGDClassifier with hinge loss**.
            Used GridSearch to tune hyperparameters.
