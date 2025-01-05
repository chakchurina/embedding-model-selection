[![ru](https://img.shields.io/badge/lang-ru-blue)](README.ru.md)

# Choosing an Embedding Model with No Dataset or Historical Data

## Introduction

With the rise of large language models, vector search has gained new momentum. Teams that implement a Retrieval-Augmented Generation (RAG) architecture face a question: how do you choose the embeddings that will work effectively with your data?

Selecting an embedding model is a strategic, long-term decision, since it directly impacts search quality and system performance. But making this choice is particularly challenging in the early stages of a project when there isn't enough data to make an informed decision. And switching models later can be costly and resource-intensive.

The solution seems simple: just check out a popular benchmark and pick a top-ranking model. But success on a leaderboard doesn’t guarantee strong performance in specialized domains like finance, healthcare, or e-commerce. Without a dataset, picking a suitable model becomes a real challenge.

In this article, we explore several approaches for evaluating embedding models even when the data is scarce. We examine aspects of vector embeddings behaviour that can guide your choice based on your project’s unique needs.

We focus on qualitative evaluation methods because, in real-world scenarios, running standard quantitative experiments isn’t always feasible. When a project lacks its golden dataset, user history, or resources for manual labeling, alternative approaches are necessary.

As an example, we analyze several leading embedding models from the MTEB leaderboard (as of this article’s writing) and demonstrate which vector representation properties—beyond leaderboard ranking—are worth considering to make an informed decision. The models for this article were selected based on the following criteria:

- Model size: We avoid large language models and opt for compact ones that are easier to deploy in production.
- License: Preference is given to models with open licenses, such as Apache 2.0.
- Vector dimensionality: To balance quality and performance, we focus on models with vector sizes up to 3,000 dimensions.

The specific models we chose aren’t the point. Our goal is to demonstrate evaluation techniques. Here’s what we used:

General-purpose models:
- [text-embedding-3-large by OpenAI](https://platform.openai.com/docs/guides/embeddings#embedding-models)
- [voyage-large-2 by VoyageAI](https://blog.voyageai.com/2024/05/05/voyage-large-2-instruct-instruction-tuned-and-rank-1-on-mteb/) 
- [gte-large-en-v1.5 by Alibaba](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)
- [jina-embeddings-v3 by JinaAI](https://huggingface.co/jinaai/jina-embeddings-v3)
- [modernbert by HuggingFace](https://huggingface.co/blog/modernbert)

Biomedical models:
- [MedEmbed-small-v0.1](https://huggingface.co/abhinand/MedEmbed-small-v0.1)
- [BioBERT-mnli-snli-scinli-scitail-mednli-stsb](https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb)

Both authors work in the pharmaceutical industry, so we use medical terms and texts for analysis. However, the evaluation methods presented here are universal and can be adapted to any domains: fintech, legaltech, e-commerce, or other industries.

Let’s dive in!

## Standard Approach to Search Quality Evaluation

Vector search typically follows these steps:

1. An embedding model generates vector representations for documents in a database.
2. A user query is converted into a vector using the same model.
3. The query vector is compared to document vectors.
4. A ranked list of the most relevant documents is returned.

In an ideal scenario, if we have

- a set of documents $D$,,
- a set of queries $Q$,
- and relevance labels mapping queries to documents $(q, d) \in Q \times D$,
we could apply standard quantitative [retrieval metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)), such as Precision@k, Recall@k, NDCG@k, or MAP.

But in the real world, the data needed to evaluate these metrics is often missing. So, let’s explore how we can obtain it.

## Finding Data for Assessment

To evaluate vector embeddings, you can use:

- Benchmarks,
- Manual annotation,
- User history.

Let’s discuss each of these sources.

### Benchmarks

Benchmarks like [MTEB](https://huggingface.co/spaces/mteb/leaderboard) provide standardized tools for evaluating embedding models. However, MTEB tests models on various tasks, like classification, clustering, text generation, and doesn’t specifically focus on vector search. As a result, top-ranking models on MTEB may not perform well in your particular use case.

An alternative is [BEIR](https://eval.ai/web/challenges/challenge-page/1897/leaderboard/4475), a platform for evaluating models in information retrieval. BEIR includes datasets for assessing different search models (not necessarily vector-based) in real-world scenarios.

However, if you check out the [BEIR dataset list](https://github.com/beir-cellar/beir), you'll notice that most datasets are geared toward:
- Medical topics,
- Question answering tasks.

Since we work in the pharmaceutical industry, this is a great starting point. But if your field is, say, legaltech, BEIR might not be as relevant.

So, while BEIR and MTEB can help shortlist candidate models, the final decision still requires further evaluation.

### Manual Annotation

Manual annotation is the most expensive and time-consuming method. It requires time, money, and continuous updates if your search system needs to adapt to evolving data.

The labeling should align with your objectives. For example, assessing search quality might involve tagging query-document pairs with relevance scores. The amount of required data depends on task complexity, and it’s crucial to account for rare edge cases that might be underrepresented in your dataset.

If you already have relevant data, make use of it. f annotation costs are unreasonably high, exploring alternative evaluation approaches might be a good idea.

### User History

Analyzing past user interactions with search results can provide valuable signals for search quality assessment. In e-commerce, this might include clicks and purchases linked to different queries. In pharma, it could be user interest in specific drugs or research topics.

However, this approach has limitations:

1. Engagement ≠ Relevance. Engaged users may interact with irrelevant results (e.g., clickbait). Conversely, highly relevant results may see low engagement, perhaps because the search worked perfectly, but users ignored the correct answers.
2. Noisy Data. For example, if a relevant item appears on the second page but the user accidentally skips it, the data won’t reflect its relevance, skewing the evaluation.
4. Rare Queries and Empty Results. For rare queries or cases where the database lacks relevant documents, it’s impossible to gather enough data for fair assessment.

These factors introduce noise and increase the risk of reinforcing existing model biases rather than mitigating them. Blindly relying on user history can be dangerous, such data has to be handled with extreme caution.

At the same time, early-stage projects may have no user history at all. In such cases, it’s especially important to use evaluation methods that don’t depend on large datasets.

## What Makes an Ideal Embedding?

What should we expect from embeddings in search task? How should an ideal model behave?

A common approach for such tasks is the two-tower architecture, which considers:
- Query-to-document matching – ensuring search results accurately reflect the query intent.
- Document-to-user matching – personalizing results based on user history and preferences.

Given this, it's essential to evaluate how well a model generates query embeddings, document embeddings, and how effectively they interact. Additionally, it might be interesting to check the model's robustness to typos, which are inevitable in real-world queries, and how well it handles domain-specific terminology.

Based on this, here are four key aspects for assessing embedding models for vector search:

- Query embeddings – How accurately does the model vectorize typical queries in your domain?
- Query-document interaction – Do query and document embeddings work well together?
- Robustness to typos – Can the model still retrieve relevant results despite user input errors?
- Handling domain-specific terms – Do vector representations effectively capture industry-specific terminology?

## Embedding Models Evaluation

### Query Embeddings

For effective search, query embeddings should capture semantic similarity. The core assumption is that embedding space should position similar queries close to each other:

*similar queries → close vectors*.

To test this, we created nine pairs of semantically close queries relevant to the medical field. In medicine, this doesn’t just mean different phrasings of the same question but also branded vs. generic drug names and abbreviations vs. full terms. Here are a few examples:

- _"Connection between LDH markers and persistent exhaustion"_ ↔ _"How are lactate dehydrogenase levels and chronic fatigue related?"_
- _"What published data is available for impact on daily activities with anti-C5 therapy?"_ ↔ _"Is there evidence on how C5i impacts patients' ability to perform daily tasks?"_
- _"Managing PNH via IVH control."_ ↔ _"Addressing paroxysmal nocturnal hemoglobinuria through the regulation of intravascular hemolysis."_

Each pair contains queries with similar meaning, but different pairs represent distinct topics.

We converted each query into a vector and calculated pairwise cosine distances between all embeddings. Since queries within the same pair should be close in meaning and queries from different pairs should be distinct, we expect the model to reflect this pattern:
- Lower distances for queries within the same pair,
- Higher distances for queries across different pairs.

The results are visualized in the heatmaps below, where color intensity represents similarity. The diagonal reflects synonymous queries, while off-diagonal values show unrelated queries.

![image](https://github.com/user-attachments/assets/b9e23cb6-8019-414a-8b50-440ed96b8a22)

**OpenAI (text-embedding-3-large):** The diagonal is clearly distinguished against the low values for queries from different pairs. The model confidently differentiates similar queries within pairs and almost entirely eliminates high scores for unrelated queries. This property is particularly valuable in applications where reducing false positives and ensuring distinct query separation is critical.

![image](https://github.com/user-attachments/assets/75637540-93c8-4fc5-a786-3d489404704d)

**Voyage (voyage-large-2):** The diagonal is poorly defined, indicating a weaker ability to differentiate between semantically similar and dissimilar queries. In some cases, unrelated queries exhibit high similarity, suggesting that the model struggles with fine-grained query discrimination.

![image](https://github.com/user-attachments/assets/d325921f-c2da-496a-8c2b-017d4bfda2ca)

**Alibaba (gte-large-en-v1.5):** The diagonal is more distinct compared to Voyage but has inconsistencies. The contrast against the background suggests the model can separate queries to some extent, though not as precisely as OpenAI. This model may be suitable for tasks where semantic flexibility is preferred over strict query separation.

![image](https://github.com/user-attachments/assets/9312c8d9-7e89-4ee6-94af-072b5a80ce2f)

**Jina (jina-embeddings-v3):** The diagonal is highly pronounced, closely resembling OpenAI's results. This suggests the model effectively separates semantically distinct queries while maintaining high similarity scores for synonymous ones.

![image](https://github.com/user-attachments/assets/70a6af89-ee52-4123-9695-e83707cd5886)

**BioBERT и MedEmbed:** Both domain-specific models perform reasonably well. BioBERT demonstrates sharper query separation, whereas MedEmbed produces softer distinction.

However, both models seem to struggle with abbreviations, as seen in the fourth query pair:
- _"Managing PNH via IVH control."_ 
- и _"Addressing paroxysmal nocturnal hemoglobinuria through the regulation of intravascular hemolysis."_

This may impact their usability in applications where accurate handling of abbreviations is essential.

![image](https://github.com/user-attachments/assets/e2876e1c-600f-4606-acbd-1dbcc283c637)

**ModernBERT:** A model can behave similar to any of the previously analyzed ones, depending on whether strict or more relaxed query separation is required. However, results should not look like ModernBERT: the lack of a visible diagonal suggests an inability to distinguish similar queries within the same domain.

The choice of model depends on the application. If strict query separation is required, OpenAI’s model is a strong candidate. For a more flexible approach, Voyage or Alibaba may be considered. The key takeaway is that the model should not behave like ModernBERT in the given visualization, where all queries appear equally relevant.

### How Query and Document Embeddings Work Together

Query and document embeddings must interact in a way that ensures accurate retrieval. To evaluate this interaction, we will need some metadata, such as category hierarchies, taxonomies, or knowledge graphs.

In biomedical data, we can use MeSH (Medical Subject Headings), a hierarchical system of medical terms. In e-commerce, a similar structure can be derived from product catalogs, in HR-tech, it may come from job title classifications. In other domains, tags, genres, or other classification schemes can serve as taxonomies.

Here's what [MeSH terms](https://meshb.nlm.nih.gov/treeView) look like

![image](https://github.com/user-attachments/assets/c10365b4-8be7-4cf2-9f3d-004806d27d0e)

For this test, we used annotations of biomedical research papers from the last five years, specifically those classified under the MeSH term _Diabetes, Gestational [C19.246.200]_ — a type of diabetes that develops during pregnancy and typically resolves after childbirth.

As a neighboring category, we selected _Latent Autoimmune Diabetes in Adults [C19.246.656]_ (LADA) —  a slow-progressing autoimmune form of diabetes diagnosed in adulthood. For LADA, we only used the category title without associated texts.

Since an average token consists of approximately ¾ of a word (i.e., 100 tokens ≈ 75 words), we excluded annotations longer than 600 words to avoid chunking, as not all models tested support long text sequences.

So, for analysis, we selected PubMed papers annotations corresponding to two clinically similar diabetes types that differ in etiology.

![image](https://github.com/user-attachments/assets/a0686398-9492-4d9f-a9cc-6a725f1affe4)

The test follows this logic: if two documents belong to the same category, their embeddings should be closer to their own category title than to the title of the neighboring one.

To quantify this, we measure the similarity of document embeddings (например, _Diabetes, Gestational_) with:
1. The title of their own category (e.g., "Gestational diabetes").
2. The title of the neighboring category (e.g., "Latent autoimmune diabetes in adults").

This evaluation helps assess:
- Whether the model differentiates between documents from distinct but related categories.
- How much variation exists in similarity scores, what are minimum, maximum, and average similarity to both categories.

![image](https://github.com/user-attachments/assets/10c7aa10-7926-40ae-90bb-f83d6a13227f)

The distributions above show the similarity scores of documents to the category titles _Gestational diabetes_ and _LADA_,  for different embedding models.

1. **OpenAI**: The average similarity of documents to their own category (Gestational diabetes) is ~0.45, while similarity to the neighboring category (LADA) is ~0.3. This indicates good category separation, with generally moderate similarity values.
    
2. **Voyage**: This model produces higher similarity scores overall—with an average of ~0.8 for both categories. While it correctly differentiates documents, overlapping distributions could pose challenges in scenarios requiring strict separation.
    
3. **Alibaba**: This model clearly distinguishes the categories, with an average similarity of ~0.8 for Gestational diabetes and just under 0.6 for _LADA_. This behavior makes Alibaba suitable for tasks requiring robust separation of semantically similar but distinct texts.
    
4. **Jina**: The results resemble Alibaba’s—the model separates categories well, though distributions are broader. The gap between the average similarity to the relevant category vs. the unrelated one suggests strong differentiation.

5. **BioBERT**: Documents show an average similarity of ~0.5 to _Gestational diabetes_ and ~0.3 to _LADA_, meaning the model correctly distinguishes categories. The model doesn't produces high similarity scores, and irrelevant documents tend to have very low values, which is consistent with its specialization in biomedical text processing.
    
6. **MedEmbed**: The model’s ability to separate categories is weaker, with average similarity scores of ~0.7 for _Gestational diabetes_ and ~0.6 for _LADA_. The distinction between categories is less pronounced compared to BioBERT, suggesting lower confidence in semantic differentiation.
    
7. **ModernBERT**: The model behaves incorrectly. Documents from _Gestational diabetes_ have a higher average similarity to _LADA_ than to their own category. The average similarity to LADA exceeds 0.8, while to Gestational diabetes it is only ~0.7. This suggests that the model fails to correctly distinguish categories.

So,  
- the selected models from Voyage, MedEmbed, and ModernBERT demonstrated insufficient ability to correctly differentiate categories in this test,  
- while the OpenAI, Alibaba, Jina, and BioBERT models performed well.

Additionally, visualizing similarity distributions helps assess the similarity values that each model assigns to domain-specific texts. These insights can be used to fine-tune threshold values for retrieval tasks, balancing recall and precision in search applications.

### Robustness to Typos

Embeddings are often perceived as a universal tool that allows a system to "understand" user intent. However, in practice, tolerance to typos heavily depends on the training data and tokenization strategy. Since users inevitably make mistakes, it's interesting to see how well a model handles them.

We constructed a list of 12 medical terms, each modified with different types of errors:

- willebrand disease → **x**illebrand disease
- ozempik → **0**zempik
- patient → **a**ptient
- pregnancy → **r**egnancy
- transfusions → trans**fys**ions
- diagnosis → dia**q**nosis
- cystinuria → cysti**un**ria
- delstrigo → delt**r**igo
- sclerosis → sclerosi**q**
- fatigue → fatigu**c**
- therapy → ther**p**ay
- cyst → cy**s**

This kind of list can be generated from system logs, using real-world typos, manually, by introducing common spelling mistakes or automatically, using tools like [typo](https://pypi.org/project/typo/) package.

To ensure a comprehensive evaluation, the dataset should include different types of errors, such as:

- Character substitutions (e.g., replacing a letter with a nearby key on the keyboard).
- Missing characters,
- Swapped characters.

Errors should appear at different positions within words: at the beginning, middle, and end, to better simulate real-world user mistakes. This approach allows for a more realistic assessment of the model's performance under noisy input conditions.

![image](https://github.com/user-attachments/assets/3f4c1553-b5c8-4594-bdfc-3bd211e8e9b5)

To evaluate typo tolerance, we measured the cosine similarity between the embeddings of original medical terms and their misspelled versions across various error types. Based on these results, we generated a heatmap all models.

1. **OpenAI** показывает стабильные результаты для большинства терминов, включая случаи "patient → aptient" (0.45) и "therapy → therpay" (0.71), которые смогли «смутить» остальные модели. Хотя значения сходства могут быть ниже, чем у других моделей, это не значит, что она не справилась. По прошлому тесту среднее сходство для релевантных документов у OpenAI было около 0.45, и так как все скоры выше среднего значения, говорит о том, что модель не считает опечатки полностью идентичными, но при этом способна находить правильную форму слова.
    
2. **Voyage** выглядит очень устойчиво к опечаткам. Например, пары "delstrigo → deltrigo" и "pregnancy → regnancy" получили почти идеальные значения (0.99). У модели высокие скоры и ни один термин не «запутал» модель, так как все скоры получились заметно выше среднего с прошлого теста (0.8).
    
3. **Alibaba** демонстрирует результаты для опечаток ниже ожидаемого уровня (0.6–0.8 для релевантных документов в прошлом тесте). Термины "fatigue" и "patient" оказались особенно сложными, но даже для остальных скоры остаются низкими на фоне среднего значения.
    
4. **Jina** в прошлом тесте показывала среднее значение для релевантных текстов около 0.6. Модель хорошо справилась с терминами "fatigue", "cystinuria" и "delstrigo", но показывает крайне низкие скоры для почти половины других терминов, включая "patient", "pregnancy" и "transfusions".
    
5. **BioBERT** последовательно выдаёт низкие значения (около 0.5 даже на релевантных данных). На этом фоне получается, что модель не распознала более половины терминов с опечатками, что указывает на крайне низкое качество обработки ошибок.

6. **MedEmbed** в прошлых тестах демонстрировала тенденцию к завышению значений, со средним скором около 0.7 для релевантных текстов. Но в этом тесте только 3 из 12 значений превысили 0.7, что свидетельствует о неустойчивости модели к опечаткам.
    
7. **ModernBERT**, со средним значением около 0.8 для релевантных текстов, выглядит относительно устойчивой моделью. В данном тесте модель пропустила только несколько терминов, что указывает на её способность справляться с опечатками. Тем не менее, неоднозначность прошлых тестов ставит под вопрос её общую адекватность.

Анализ устойчивости к опечаткам показывает, насколько по-разному модели справляются с пользовательскими ошибками. Такая оценка помогает лучше понять, как модель взаимодействует с некорректными данными и насколько надёжно она будет работать в реальных сценариях. Например, **Alibaba**, **Jina** и **BioBERT**, которые показали уверенные результаты на предыдущих тестах, продемонстрировали слабые результаты в этом анализе. Это подчеркивает важность комплексного подхода к выбору модели: одна и та же модель может быть сильной в одних задачах и уязвимой в других.

todo: дополнительно можно обратить внимание на токенизацию, так как она определяет, как модель обрабатывает ошибочные слова. Сравните, как наши модели токенизируют слова с опечатками и без.

### Работа с доменными терминами

Для оценки качества эмбеддингов в специализированной области важно понять, насколько корректно модель обрабатывает термины предметной области. Если модель не распознаёт контекст и не связывает термины с их синонимами или близкими понятиями, это может привести к потере релевантной информации. Для качественной оценки того, как модель работает с терминами, сделаем следующее: 

1. Соберем набор специализированных терминов из медицинской области, включая как общеизвестные термины, так и редкие. Например:
    - _Metformin_ — популярный препарат для лечения диабета.
    - _Waldenström Macroglobulinemia_ — редкая форма лимфомы.
    - _lncRNA_ — длинные некодирующие РНК, связанные с регуляцией генов.
2. Преобразуем их в эмбеддинг-векторы, используя модели, участвующие в анализе.
3. Сравним полученные векторы с векторами из какого-нибудь доступного словаря (например, WordNet), чтобы определить ближайшие по смыслу слова и таким образом увидеть, «понимает» ли модель термины нашего домена.

Если ближайшие к термину векторы представляют синонимы, связанные концепции или термины из той же области, это говорит о том, что модель корректно улавливает контекст. Например, если к «РНК» модель относит слова «геном» или «рибосома», это указывает на её способность корректно работать с этим термином. Если же модель выберет из словаря слоги или не связанные с медициной слова, это сигнализирует о проблемах в работе модели с терминами.

Ниже результаты для разных моделей:

| **Термин**                        | **OpenAI**                                        | **Voyage**                                           | **Alibaba**                                        | **Jina**                                           | **BioBERT**                                      | **MedEmbed**                                         | **ModernBERT**                                 |
| --------------------------------- | ------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------- | ---------------------------------------------- |
| **lncRNA**                        | nrna, nuclear_rna, informational_rna              | nuclear_rna, mrna, rna                               | rna, informational_rna, mrna                       | nrna, rna, nrl                                     | lunda, livistona, liliales                       | rna, nuclear_rna, nrna                               | mycophage, chalcid, flecainide                 |
| **BBB disruption therapy**        | blood-brain_barrier, thrombolytic_therapy, clot   | therapeutic, therapy, therapeutical                  | implosion_therapy, disrupt, therapeutical          | implosion_therapy, behavior_therapy, disruption    | bobsledding, bd, boding                          | disruption, bb, bbs                                  | feedlot, birthwort, bombastically              |
| **Antisense oligonucleotide**     | nucleoside, antimetabolite, dideoxyinosine        | antibody, recombinant, isoantibody                   | nucleoside, didanosine, mrna                       | non-nucleoside_reverse_transcriptase_inhibitor     | ribonucleic_acid, antineoplastic, knock-down     | nucleotide, dna, endonuclease                        | queenfish, immensurable, antihistamine         |
| **PD-L1 mAbs**                    | cancer_drug, immunotherapeutic, anti-tnf_compound | monoclonal, monoclonal_antibody, antibody            | monoclonal_antibody, immunotherapeutic, monoclonal | immunoglobulin_d, monoclonal_antibody, immunoassay | mam, lgb, mamma                                  | monoclonal_antibody, monoclonal, pd                  | l-plate, hsv-2, 401-k_plan                     |
| **Waldenström Macroglobulinemia** | plasmacytoma, myeloma, gammopathy                 | agammaglobulinemia, hypogammaglobulinemia, granuloma | agammaglobulinemia, multiple_myeloma, myeloma      | agammaglobulinemia, myoglobinuria, megaloblastoma  | myoglobinuria, mulishness, hypogammaglobulinemia | agammaglobulinemia, waldenses, hypogammaglobulinemia | osteosclerosis_congenita, drepanocytic_anaemia |
| **Frey syndrome**                 | hyperhidrosis, diaphoresis, polyhidrosis          | syndrome, williams_syndrome, idiopathy               | frey, bruxism, waterhouse-friderichsen_syndrome    | syndrome, frey, reye's_syndrome                    | frey, freyr, freya                               | frey, freya, freyr                                   | frostwort, foumart, fortunella                 |
| **Ozempic**                       | glucophage, glipizide, zapotecan                  | otic, zocor, empirin                                 | ozena, obidoxime_chloride, oxyphencyclimine        | ocimum, omotic, oecumenic                          | ozena, ozarks, ozaena                            | ozonize, typic, ozonide                              | palometa, pseudemys, empyreal                  |
| **Cladribine**                    | chlorambucil, leukeran, mercaptopurine            | clonidine, chlorpromazine, clomipramine              | zalcitabine, lamivudine, deoxyadenosine            | cladrastis, clerid, clinid                         | cladonia, cladrastis, cladode                    | clad, cladode, zalcitabine                           | cladoniaceae, cladophyll, cicadellidae         |
| **Zolgensma**                     | zoloft, lofortyx, vincristine                     | zetland, zinzendorf, nijmegen                        | zeugma, zygnema, genus_zygnema                     | zola, zymogen, zolaesque                           | ziziphus, zola, zetland                          | zeugma, glechoma, zygoma                             | schmaltz, spotweld, zinfandel                  |
| **ReoPro**                        | lipo-hepin, thrombolytic_agent, plavix            | pro, re, ream                                        | appro, reechoing, recopy                           | pro, recco, ro                                     | reproval, retem, reamer                          | requital, reenact, reproof                           | galactocele, rectocele, proviso                |

Таблица показывает, что не все модели хорошо обрабатывают медицинские термины. 

**OpenAI** продемонстрировала отличные результаты. Она единственная корректно связала термин **BBB disruption therapy** с _blood-brain_barrier_ из WordNet. Термин **PD-L1 mAbs** правильно ассоциировала с _cancer_drug_, отражая связь с лекарствами от рака. Точно обработала **Frey syndrome**, связав его с _hyperhidrosis_, что соответствует клиническому проявлению этого синдрома. Также успешно определила кластер лекарств для **Ozempic**, связав его другими медикаментами для лечения диабета второго типа. Модель не справилась с терминами **Zolgensma** и **ReoPro**, но сохранила ассоциацию с медицинским контекстом.

**Voyage** продемонстрировала средние результаты. Для **BBB disruption therapy** модель зацепилась за слово _therapy_, но не смогла вытащить релевантное значение. Для **Frey syndrome** она извлекла лишь общие термины со словом «синдром», не учитывая специфики. В **PD-L1 mAbs** модель правильно распознала сокращение, но не ассоциировала его механизмом действия или назначением. Для **Ozempic** и **ReoPro** модель не смогла выделить ничего значимого, вытащив токены, такие как _otic_, _zocor_, _ream_. В целом, Voyage справляется, но обрабатывает медицинские термины поверхностно.

**Alibaba** проявила себя неоднородно. Например, она ассоциировала **BBB disruption therapy** с психотерапевтическими терминами (_implosion_therapy_), что является ошибкой. С **Antisense oligonucleotide** модель справилась неплохо, распознав молекулярные компоненты, и с **Waldenström Macroglobulinemia** — определив её как опухоль. Однако с **Ozempic** модель не смогла выдать полезные ассоциации, предложив термины из другого домена. **Zolgensma** также осталась для неё загадкой, так как среди ближайших слов оказались _zeugma_ и _zygnema_, не связанные с медициной.

**Jina** явно не ориентирована на медицинский контекст. Например, для терминов **lncRNA**, **Ozempic**, **Cladribine**, **Zolgensma** и **ReoPro** она предложила слоги вместо осмысленных ассоциаций. С **BBB disruption therapy** модель также ошибочно вытащила психотерапевтические термины (_implosion_therapy_). Несмотря на это, она справилась с терминами вроде **Antisense oligonucleotide**, **PD-L1 mAbs**, и **Waldenström Macroglobulinemia**, значительно уступая OpenAI и Voyage.

**BioBERT**, неожиданно для модели, обученной на медицинских данных, показал удивительно слабые результаты. Она не смогла корректно обработать большинство терминов, включая **BBB disruption therapy**, где выдала совершенно нерелевантное _bobsledding_. Видна тенденция к опоре на отдельные токены, например, для **Ozempic** и других лекарств. Для **Frey syndrome** модель предложила набор нерелевантных слов (_frey_, _freyr_, _freya_), что указывает на её слабость в медицинском домене.

**MedEmbed** оказалась только чуть лучше BioBERT. Она сохраняет общую привязку к медицинскому домену, что делает её использование в этом контексте возможным, хотя с лекарствами, такими как **Ozempic** и **Cladribine**, модель не справилась.

**ModernBERT** просто не работает: её ассоциации в более, чем в половине случаев вообще выходят за рамки медицинского домена. Например, вместо медицинских терминов модель предлагает слова вроде bombastically, queenfish, frostwort, schmaltz. Это указывает на то, что модель на практике не справляется с обработкой медицинского контекста.

Анализ работы с доменными терминами показал, что эффективность моделей в специализированных областях может значительно варьироваться. Этот тест демонстрирует, что заявленные способности модели или место на лидерборде не всегда соответствуют её реальной эффективности, особенно в задачах, требующих глубокого понимания специфики домена.

Оптимальная модель должна улавливать контекст, корректно ассоциировать термины с их синонимами и близкими понятиями, обеспечивая высокое качество поиска. Но даже лучшие модели имеют ограничения: новые или редко встречающиеся термины, не вошедшие в обучающие данные, останутся для них недоступными. В таких случаях комбинированный подход, включающий эмбеддинги и полнотекстовый поиск, может повысить качество системы.

## Бонус трек: поиск выбросов 

Если вы дочитали до этого места, то это бонус-трек:) 

Что модель считает «противоположностью» вашему запросу? Если у вас уже есть работающая поисковая система, попробуйте вместо top-n документов запросить те, которые находятся в самом конце списка по косинусному сходству .

Это позволит увидеть, какие данные модель считает минимально релевантными. Если эмбеддинги работают корректно, вы можете обнаружить выбросы — документы, которые сильно отличаются от остальных. Возможно, такие данные не должны были попасть в базу или нуждаются в отдельной обработке.

Таким способом мы случайно обнаружили на одном из проектов в базе медицинских текстов:
- документы на японском, 
- презентацию с корпоративных мероприятий,
- описание случайного ресторана в Португалии.

Это своеобразный sanity check на больших объемах данных, который может не только дать вам представление о поведении модели, но и помочь почистить данные. К тому же, это весело!

## Вывод

Представленные методы анализа эмбеддинг-моделей позволяют глубже понять их сильные и слабые стороны. Эти подходы не только просто реализовать, но и легко объяснить бизнесу или клиентам. Они помогают быстро получить ценные инсайты об эмбеддинг-моделях, даже если полноценный датасет для тестирования отсутствует.

Эти тесты также показывают, что как бы хороши не были ваши эмбеддинги, если ваш проект связан со сложным или быстро развивающимся доменом, то стоит рассмотреть гибридные подходы, которые включают и векторный, и полнотекстовый поиск. Полнотекстовый поиск может стать отличной опорой для работы с новыми или редко встречающимися терминами, которые ещё не попали в данные, на которых обучали модели.

Методы, описанные в статье, показывают, что даже при ограниченных ресурсах можно сделать информированный выбор векторной модели для вашего проекта, чтобы оптимизировать вашу поисковую систему, делая их её более точной, устойчивой к ошибкам и удобной для пользователей.

Спасибо, что дочитали статью до конца! Если у вас есть вопросы или идеи, будем рады обсудить их в комментариях.

Заглядывайте на наш [GitHub](https://chatgpt.com/c/6771b12c-248c-8001-a866-413704195300#): мы оставили там блокнот, который вы можете адаптировать для своих задач. Будем благодарны за звёздочки, пулл-реквесты и комментарии!

Если вашему бизнесу или команде нужна экспертиза в обработке текстов, пишите в LinkedIn [Марии](https://www.linkedin.com/in/maria-chakchurina/) или [Екатерине](https://www.linkedin.com/in/ekaterina-antonova-51108a67/) — обсудим ваши задачи и поможем с их решением. А еще подписывайтесь, чтобы не пропустить новые статьи!

## Вдохновлено

- [https://haystackconf.com/eu2023/talk-13/](https://haystackconf.com/eu2023/talk-13/)
- https://haystackconf.com/eu2023/talk-7/ 
