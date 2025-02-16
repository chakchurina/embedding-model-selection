[![ru](https://img.shields.io/badge/lang-ru-blue)](README.ru.md)

# Choosing an Embedding Model with No Dataset or Historical Data

## Introduction

With the rise of large language models, vector search has gained new momentum. Teams that implement a Retrieval-Augmented Generation (RAG) architecture face a question: how do you choose the embeddings that will work effectively with your data?

Selecting an embedding model is a strategic, long-term decision, since it directly impacts search quality and system performance. But making this choice is particularly challenging in the early stages of a project when there isn't enough data to make an informed decision. And switching models later can be costly and resource-intensive.

The solution seems simple: just check out a popular benchmark and pick a top-ranking model. But success on a leaderboard doesn’t guarantee strong performance in specialized domains like finance, healthcare, or e-commerce. Without a dataset, picking a suitable model becomes a real challenge.

In this article, we explore several approaches for evaluating embedding models even when the data is scarce, particularly in domain-specific applications. We examine aspects of vector embeddings behaviour that can guide your choice based on your project’s unique needs.

This article focuses on qualitative evaluation methods for embeddings since, in real-world conditions, it is often impossible to conduct standard quantitative experiments. When a project lacks a golden dataset, user history, or resources for manual annotation, alternative assessment strategies are necessary.

As an example, we analyze several leading embedding models from the MTEB leaderboard (as of the time of writing) and demonstrate which embeddings properties—beyond leaderboard ranking—are worth considering to make an informed decision. The models for this article were selected based on the following criteria:

- Model size: We avoid large language models and opt for compact ones that are easier to deploy in production.
- Vector dimensionality: To balance quality and performance, we focus on models with vector sizes up to 3,000 dimensions.
- Domain specificity: To test how well embeddings handle specialized terminology, we include several biomedical models in the evaluation.

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

We intentionally included ModernBERT, a base language model that has not been trained for sentence similarity tasks. Here it serves as a negative example, illustrating behaviour of  unsuitable model. By the way, kudos to the model’s creators—it is an impressive achievement.

Both authors work in the biomedical data, so we use medical terms and texts for analysis. However, the evaluation methods presented here are universal and can be adapted for any specialized field: whether fintech, legaltech, e-commerce, or other industries.

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

Thus, while BEIR and MTEB can help shortlist candidate models, the final decision still requires further evaluation.

### Manual Annotation

Manual annotation is the most expensive and time-consuming method. It requires time, money, and continuous updates if your search system needs to adapt to evolving data.

The labeling should align with your objectives. For example, assessing search quality might involve tagging query-document pairs with relevance scores. The amount of required data depends on task complexity, and it’s crucial to account for rare edge cases that might be underrepresented in your dataset.

If you already have relevant data, make use of it. But if annotation costs are unreasonably high, exploring alternative evaluation approaches might be a good idea.

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

In this pair, one query uses a standard medical abbreviation, while the other presents the full term:
- "PTT abnormalities in individuals with lupus anticoagulant"
- "Partial thromboplastin time variations in patients with antiphospholipid syndrome"

Here, the difference is whether a biological term is abbreviated or fully written out:
- "Role of M1dG in oxidative DNA damage"
- "Impact of malondialdehyde-deoxyguanine adducts on genomic stability"

This pair demonstrates synonyms in use:
- "Earwax removal is sometimes necessary when natural cleaning mechanisms fail."
- "When cerumen does not clear on its own, medical intervention may be required."

Each pair contains queries with similar meaning, but different pairs represent distinct topics.

We converted each query into a vector and calculated pairwise cosine distances between all embeddings. Since queries within the same pair should be close in meaning and queries from different pairs should be distinct, we expect the model to reflect this pattern:
- Lower distances for queries within the same pair,
- Higher distances for queries across different pairs.

The results are visualized in the heatmaps below, where color intensity represents similarity. The diagonal reflects synonymous queries, while off-diagonal values show unrelated queries.

![image](https://github.com/user-attachments/assets/970c603b-62a3-4431-9ca8-1c3fec0181f4)

**OpenAI (text-embedding-3-large):** The diagonal is clearly distinguished against the low values for queries from different pairs. The model confidently differentiates similar queries within pairs and almost entirely eliminates high scores for unrelated queries. This property is particularly valuable in applications where reducing false positives and ensuring distinct query separation is critical.

![image](https://github.com/user-attachments/assets/036ea562-e3dc-4e6a-b009-b0d7c37e9541)

**Voyage (voyage-large-2):** The diagonal is poorly defined, indicating a weaker ability to differentiate between semantically similar and dissimilar queries. In some cases, unrelated queries exhibit high similarity, suggesting that the model struggles with fine-grained query discrimination.

![image](https://github.com/user-attachments/assets/c9e01cc9-da40-4a41-94cd-1c39ec57a487)

**Alibaba (gte-large-en-v1.5):** The diagonal is more distinct compared to Voyage but has inconsistencies. The contrast against the background suggests the model can separate queries to some extent, though not as precisely as OpenAI. This model may be suitable for tasks where semantic flexibility is preferred over strict query separation.

![image](https://github.com/user-attachments/assets/19ee49e3-3e60-4d87-9641-22508d735035)

**Jina (jina-embeddings-v3):** The diagonal is highly pronounced, closely resembling OpenAI's results. This suggests the model effectively separates semantically distinct queries while maintaining high similarity scores for synonymous ones.

![image](https://github.com/user-attachments/assets/c146ddc2-e069-4e0d-98f7-6c9f18f92bbf)

**BioBERT and MedEmbed:** Both domain-specific models perform reasonably well. BioBERT demonstrates sharper query separation, whereas MedEmbed produces softer distinction. However, both models seem to struggle with abbreviations, which may impact their usability in applications where accurate handling of abbreviations is essential.

![image](https://github.com/user-attachments/assets/39f199ee-7e96-42e7-b2ad-d0b7a9e54cbd)

**ModernBERT-gte (gte-modernbert-base):** performs well in this test: the diagonal is generally well-defined, with no high scores outside of it. Off-diagonal values are relatively high compared to the diagonal, meaning the model softly separates non-synonymous queries that still belong to the same domain.

![image](https://github.com/user-attachments/assets/baebc5b6-c21a-4400-bd1d-4ddd6a883dd0)

**ModernBERT:** A model can behave similar to any of the previously analyzed ones, depending on whether strict or more relaxed query separation is required. However, results should not look like ModernBERT: the lack of a visible diagonal suggests an inability to distinguish similar queries within the same domain.

The choice of model depends on the application. If strict query separation is required, OpenAI’s model is a strong candidate. For a more flexible approach, Voyage or ModernBERT-gte may be considered. The key takeaway is that the model should not behave like ModernBERT in the given visualization, where all queries appear equally relevant.

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

To quantify this, we measure the similarity of document embeddings (eg., _Diabetes, Gestational_) with:
1. The title of their own category (e.g., "Gestational diabetes").
2. The title of the neighboring category (e.g., "Latent autoimmune diabetes in adults").

This evaluation helps assess:
- Whether the model differentiates between documents from distinct but related categories.
- How much variation exists in similarity scores, what are minimum, maximum, and average similarity to both categories.

![image](https://github.com/user-attachments/assets/8d075812-3be7-4ad8-ab3a-d88f16a7b5c9)

The distributions above show the similarity scores of documents to the category titles _Gestational diabetes_ and _LADA_,  for different embedding models.

1. **OpenAI (text-embedding-3-large)**: The average similarity of documents to their own category (Gestational diabetes) is ~0.45, while similarity to the neighboring category (LADA) is ~0.3. This indicates good category separation, with generally moderate similarity values.
    
2. **Voyage (voyage-large-2)**: This model produces higher similarity scores overall—with an average of ~0.8 for both categories. While it correctly differentiates documents, overlapping distributions could pose challenges in scenarios requiring strict separation.
    
3. **Alibaba (gte-large-en-v1.5)**: This model clearly distinguishes the categories, with an average similarity of ~0.8 for Gestational diabetes and just under 0.6 for _LADA_. This behavior makes Alibaba suitable for tasks requiring robust separation of semantically similar but distinct texts.
    
4. **Jina (jina-embeddings-v3)**: The results resemble Alibaba’s—the model separates categories well, though distributions are broader. The gap between the average similarity to the relevant category vs. the unrelated one suggests strong differentiation.

5. **BioBERT**: Documents show an average similarity of ~0.5 to _Gestational diabetes_ and ~0.3 to _LADA_, meaning the model correctly distinguishes categories. The model doesn't produces high similarity scores, and irrelevant documents tend to have very low values, which is consistent with its specialization in biomedical text processing.
    
6. **MedEmbed**: The model’s ability to separate categories is weaker, with average similarity scores of ~0.7 for _Gestational diabetes_ and ~0.6 for _LADA_. The distinction between categories is less pronounced compared to BioBERT, suggesting lower confidence in semantic differentiation.

7. **ModernBERT-gte (gte-modernbert-base):** The average similarity of texts to their own category is around 0.7, while to the neighboring category, it is approximately 0.6, with minimal overlap between distributions. The model tends to assign high similarity scores, but confidently distinguishes relevant texts from non-relevant ones, and performs well in this test.

8. **ModernBERT**: The model behaves incorrectly. Documents from _Gestational diabetes_ have a higher average similarity to _LADA_ than to their own category. The average similarity to LADA exceeds 0.8, while to Gestational diabetes it is only ~0.7. This suggests that the model fails to correctly distinguish categories.

Мisualizing similarity distributionss helps assess how well a model handles texts in your domain and provides insights into the similarity values typical for each model. It reveals the cosine distance range between documents and queries while also offering intuition about the shape of the distribution. These insights can guide the selection of appropriate threshold values for document filtering in retrieval tasks, helping you balance recall and precision in search applications.

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

1. **OpenAI** (text-embedding-3-large) demonstrates consistent performance across most terms, including difficult cases like "patient → aptient" (0.45) and "therapy → therpay" (0.71), which confused other models. While its similarity scores are lower than those of some other models, this does not indicate failure. In the previous test, OpenAI's average similarity for relevant documents was ~0.45. Since all typo-induced similarities remain above this threshold, the model appears to distinguish typos from fully correct terms while still recognizing their intended meaning.
    
2. **Voyage** (voyage-large-2) appears to be highly robust to typos. For example, "delstrigo → deltrigo" and "pregnancy → regnancy" received near-perfect similarity scores (~0.99). The model consistently produced high similarity values, and no term "confused" it, as all scores were significantly higher than the previous test’s average (~0.8).
    
3. **Alibaba** (gte-large-en-v1.5) demonstrates lower-than-expected performance on typos (0.6–0.8 for relevant documents in earlier tests). Words like "fatigue" and "patient" look particularly problematic. Even for other terms, similarity scores remained below the expected range, suggesting that the model struggles with misspelled input.
    
4. **Jina** (jina-embeddings-v3) which previously showed an average similarity of ~0.6 for relevant texts, handled words like "fatigue", "cystinuria", and "delstrigo" well. However, it produced extremely low similarity scores for nearly half of the terms, including "patient", "pregnancy", and "transfusions".
    
5. **BioBERT** consistently produces low similarity scores (with an average of ~0.5 on relevant data from the previous test). Given this, the model failed to recognize more than half of the typo-modified terms, indicating poor error-handling capabilities.

6. **MedEmbed** previously tended to assign high similarity scores to relevant data, with an average of ~0.7 for relevant texts. However, in this test, only 3 out of 12 terms exceeded 0.7, indicating a lack of robustness to typos.
   
7. **ModernBERT-gte** (gte-modernbert-base): in the previous test, it showed an average similarity of around 0.7 for relevant texts, and here the model produces comparable results. Most typo-modified terms received scores consistent with expected values for relevant pairs. The model only failed to handle the typo in "therapy", but in all other cases, it confidently recognized misspelled words.
    
8. **ModernBERT**, with an average similarity of ~0.8 for relevant texts, appears relatively resistant to typos. It missed only a few terms, suggesting strong error tolerance. However, inconsistencies observed in previous tests raise concerns about the model’s overall reliability.

This analysis shows how differently models handle user input errors. Understanding typo tolerance is crucial for assessing how a model processes noisy data and how reliably it performs in real-world search scenarios. This highlights the importance of a holistic model selection approach, since a model that excels in one aspect may appear vulnerable in another.

### Handling Domain-Specific Terms

When evaluating embedding quality in a specialized field, it is crucial to understand how well a model processes domain-specific terminology. If a model fails in this specific context and does not associate terms with their synonyms or related concepts, it can lead to a loss of relevant information in retrieval tasks. To assess how well models handle domain-specific terms, we follow these steps:

1. Curate a set of specialized medical terms, including both well-known and rare ones (medications, therapeutic agents, diseases, etc). Our list:
    - _lncRNA_ — long non-coding RNAs with gene regulatory functions.  
    - _BBB disruption therapy_ — method of temporarily disrupting the blood-brain barrier to deliver drugs to brain tissue.
    - _Ozempic_ — drug for treating Type 2 diabetes and weight management. 
    - _Antisense oligonucleotide_ — short synthetic DNA or RNA used in therapy for genetic diseases and cancer.
    - _PD-L1 mAbs_ — monoclonal antibodies that help the immune system recognize and destroy cancer cells.  
    - _Kabuki syndrome_ — rare genetic disorder.  
    - _Waldenström Macroglobulinemia_ — rare type of non-Hodgkin lymphoma.  
    - _Frey syndrome_ — condition causing facial sweating when eating.  
    - _Metformin_ — medication for Type 2 diabetes treatment (brand name Glucophage).   
    - _Cladribine_ — drug for treating multiple sclerosis.  
    - _Zolgensma_ — gene therapy for spinal muscular atrophy.  
    - _ReoPro_ — drug preventing blood clotting during vascular surgeries.
2. Convert these terms into embedding vectors using the models under evaluation.
3. Compare the resulting vectors against a reference dataset (e.g., WordNet) to determine the nearest semantic neighbors and evaluate whether the model "understands" domain-specific terms.

If the nearest vectors correspond to synonyms, related concepts, or terms from the same field, it indicates that the model correctly captures context. For example, if "RNA" is associated with "genome" or "ribosome," this suggests that the model accurately understands the term. However, if the model associates medical terms with random syllables or unrelated words, it signals weaknesses in handling specialized vocabulary.

The table below demonstrates that not all models handle medical terms effectively.

| **Term**                        | **OpenAI**                                        | **Voyage**                                           | **Alibaba**                                        | **Jina**                                           | **BioBERT**                                      | **MedEmbed**                                         | **ModernBERT**                                 |
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

1. **OpenAI (text-embedding-3-large)** performed exceptionally well. For instance, it was the only model that correctly linked term "BBB disruption therapy" to "blood-brain barrier" in WordNet. It also properly associated "PD-L1 mAbs" with cancer drug, capturing its relevance to oncology treatments. For "Frey syndrome", it identified "hyperhidrosis", which matches the syndrome’s clinical manifestation. It also correctly clustered medications related to "Ozempic", grouping it with other type 2 diabetes treatments. However, the model struggled with "Zolgensma" and "ReoPro", though it still retained a general medical context in its associations.

2. **Voyage (voyage-large-2)** delivered moderate performance. For "BBB disruption therapy", it latched onto the word "therapy" but failed to extract its relevant meaning. For "Frey syndrome", it retrieved generic terms containing “syndrome” without capturing its specificity. While "PD-L1 mAbs" was correctly identified as an abbreviation, the model did not link it to its mechanism of action or therapeutic use. "Ozempic" and "ReoPro" were poorly handled, producing token-based outputs such as "otic", "zocor", and "ream". Overall, Voyage provides surface-level processing of medical terms without deep semantic understanding.

3. **Alibaba (gte-large-en-v1.5)** showed inconsistent performance. For instance, it associated "BBB disruption therapy" with psychotherapy-related terms (like "implosion therapy"), which is incorrect. However, it correctly recognized "Antisense oligonucleotide" by identifying molecular components and linked "Waldenström Macroglobulinemia" to tumor-related terms. That said, "Ozempic" failed to yield meaningful associations, and "Zolgensma" remained unrecognized, with nearest terms including "zeugma" and "zygnema", which are unrelated to medicine.

4. **Jina Jina (jina-embeddings-v3)** clearly lacks a medical domain focus. It produced nonsensical syllabic outputs for terms like "lncRNA", "Ozempic", "Cladribine", "Zolgensma", and "ReoPro" instead of meaningful associations. For "BBB disruption therapy", it incorrectly retrieved psychotherapy-related terms ("implosion therapy"). However, it performed better on terms like "Antisense oligonucleotide", "PD-L1 mAbs", and "Waldenström Macroglobulinemia", though it still significantly underperformed compared to OpenAI and Voyage.

5. **BioBERT**, despite being a biomedical model, performed poorly. It failed to correctly process most terms, including "BBB disruption therapy", where it returned "bobsledding". The model displayed a strong bias towards token-based retrieval, failing to generalize term meanings. For "Frey syndrome", it returned words like "frey", "freyr", and "freya", which have no semantic relevance, indicating weak medical domain alignment.

6. **MedEmbed** performed only slightly better than BioBERT. It maintained a general connection to the medical field, making it somewhat usable in this context. However, it failed to handle pharmaceutical terms such as "Ozempic" and "Cladribine:, producing inadequate associations.

7. **ModernBERT** is entirely unreliable for medical terminology. More than half of its associations fell outside the medical domain. Instead of relevant terms, the model returned words like "bombastically", "queenfish", "frostwort", and "schmaltz". This indicates that the model is not suitable for handling medical context.

The analysis of domain-specific terms reveals significant variation in model performance for specialized fields. This test underscores that leaderboard rankings or general-purpose claims do not always reflect a model’s real-world effectiveness, especially for domain-specific applications.

An ideal model should capture context accurately, associate terms with their synonyms and conceptually related words to nsure high-quality retrieval in specialized fields. However, even the best models have limitations: new or rare terms absent in training data will not be recognized. In such cases, a hybrid approach combining embeddings with full-text search can enhance system performance and improve search accuracy.

## Bonus Track: Outlier Detection

If you've made it this far, here's a bonus track!

What does your model consider the "opposite" of your query? If you already have a working search system, try retrieving the least relevant documents based on cosine similarity instead of the top-n results.

This technique helps identify what the model deems minimally relevant, which can reveal outliers, documents that significantly differ from the rest. Such data points may require further investigation or additional preprocessing to ensure they belong in your index.



This technique serves as a sanity check for large-scale datasets, providing insights into model behavior while helping improve data quality. And honestly? It’s just fun!

## Conclusion

The methods presented in this article provide a deeper understanding of an embedding model’s strengths and weaknesses. These approaches are not only easy to implement but also straightforward to communicate to business stakeholders and clients. They offer insights into embedding models, even if a fully labeled test dataset is unavailable.

These tests also highlight a critical point: no matter how good your embeddings are, if your project operates in a complex or rapidly evolving domain, hybrid approaches should be considered. Combining vector search with full-text search can significantly enhance retrieval performance, especially for new or rare terms that haven’t appeared in the model’s training data.

The techniques discussed here demonstrate that even with limited resources, you can make an informed choice when selecting an embedding model for your project. Optimizing your search system with the right embeddings can make it more accurate, robust, and user-friendly.

Thank you for reading! If you have questions or ideas, feel free to share them in the comments—we’d love to discuss!

Check out our [GitHub](https://github.com/chakchurina/embedding-model-selection/): where we’ve shared a notebook that you can adapt to your own use cases. We appreciate stars, pull requests, and feedback!

If your business or team needs expertise in text processing, reach out to [Maria](https://www.linkedin.com/in/maria-chakchurina/) or [Ekaterina](https://www.linkedin.com/in/ekaterina-antonova-51108a67/) on LinkedIn — we’d be happy to discuss your needs and help with solutions. Also, follow us so you don’t miss future articles!

## Inspired by

- https://haystackconf.com/eu2023/talk-13/
- https://haystackconf.com/eu2023/talk-7/ 
