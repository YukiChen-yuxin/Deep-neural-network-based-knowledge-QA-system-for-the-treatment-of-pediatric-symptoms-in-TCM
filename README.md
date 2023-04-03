# Deep-neural-network-based-knowledge-QA-system-for-the-treatment-of-pediatric-symptoms-in-TCM :star::star::star::star::star::star::star::star:
Constructing a corpus of ancient Chinese pediatric medicine literature, using algorithms such as BERT, lattice LSTM, and Siamese for tasks such as named entity recognition, intent recognition, entity similarity calculation, and entity linking, to develop a TCM-QA.

## Introduction
With the wide application of deep learning technology in the field of natural language processing, question answering systems have also achieved vigorous development. The question answering system uses natural language as the interaction method and returns concise and accurate matching answers by searching corpus, knowledge graph or question answering knowledge base. Compared with search engines, the question answering system can better understand the real intention of the user's question, and can further meet the user's information needs more effectively. Question answering system is a research direction in the field of artificial intelligence and natural language processing that has attracted much attention and has broad prospects for development.
As a traditional Chinese medicine, traditional Chinese medicine (TCM) has accumulated a large number of medical practice theories. How to quickly obtain the required knowledge from it is also an important research issue in the field of TCM informatization. Aiming at the field of pediatric symptom therapy, this paper adopts the method of deep neural network to realize the knowledge question and answer system of Chinese medicine pediatric symptom therapy. The specific research includes the following contents:

(1) Construction of a corpus of ancient Chinese medicine pediatrics. This paper firstly preprocesses ancient Chinese medicine books such as "Qianjin Fang", "Treatise on Febrile Diseases", "Traditional Chinese Medicine Dictionary", "Puji Fang", extracts the data related to pediatrics, combines the medical competition database, and uses the data enhancement method of back-translation, and finally obtains There are 7,679 symptom data, 44,129 named entity identification data, 721 intent identification data, and 12,325 entity link data.

(2) TCM Named Entity Recognition. In view of the fact that traditional named entity recognition methods cannot make full use of information, and word segmentation errors will lead to problems such as error propagation, this paper uses the Lattice LSTM model to label entities by combining the information of characters, character two-tuples and words, and finally constrains the result labels through the CRF layer. , and output the named entity recognition result, which can make better use of the information in the question sentence and improve the accuracy of entity recognition. The final model recognition accuracy rate is 96.96%.

(3) Interrogation intent recognition based on Bert model. Aiming at the problem of insufficient semantic feature extraction, this paper uses the Bert model to identify the intent of the question sentence. Its attention mechanism can allocate limited information processing resources to important parts, and finally the F1 values of each intent reach 71.79%,85.19%, 94.12% and 95.24%, the model accuracy rate is 87.50%.

(4) Entity similarity calculation and answer retrieval based on knowledge graph. The LSTM-based twin neural network model is used to calculate the similarity between the entity in the question and the candidate entity in the knowledge graph, and the threshold is set for entity mapping, and the accuracy rate reaches 89.86%. 

Finally, the corresponding entities and intents are converted into query language, and the retrieved answer is filled in the answer template and returned to the user.
Experiments show that, based on the knowledge in the field of traditional Chinese medicine and pediatrics, this paper can more completely realize the application process from corpus construction to algorithm design to question and answer retrieval, and basically realize more accurate answers to users' natural language questions.

## Tips
There are some files too large to upload, email me (yuxin.yuki.chen@gmail.com) to get all files.

## How to Contribute
Please note that this project is released with a [Contributor Code of Conduct](/CODE_OF_CONDUCT.md).
By participating in this project you agree to abide by its terms.              
         
If you think you can help in any of these areas or in many areas we haven't thought of yet, then please take a look at our [Contributors' guidelines](/CONTRIBUTING.md).          
           
## Contact us
If you want to report a problem or suggest an enhancement we'd love for you to [open an issue](../../issues) at this github repository because then we can get right on it. But you can also contact [Yuki(Yuxin)](https://github.com/YukiChen-yuxin) by email yuxin.yuki.chen@gmail.com.
