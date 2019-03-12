| **Category**  | **Project**                                                                                                  | **Description**                                                                                                                                                                 |
|---------------|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **NLP**       | Toxic Comments Classification (RNN-LSTM/GRU) Metric: **AUC ROC** Models: **LogisticRegression RNN-LSTM/GRU** | Used Matplotlib and Searborn to visualize distribution of target classes, correlation of categorical variables using confusion mtrix and frequently used words using WordCloud. |
| col 2 is      | centered                                                                                                     | \$12                                                                                                                                                                            |
| zebra stripes | are neat                                                                                                     | \$1                                                                                                                                                                             |

1.  Created more features and visualized those features to understand whether
    longer comments are more toxic, etc

2.  Vectorized the words using TF-IDF and used **LogisticRegression** with
    additional features created from step 2 and without those features.

3.  Corrected spelling mistakes using **FastText embeddings** and trained a
    **RNN** network with **LSTM** and **GRU** cells using **Keras**.
