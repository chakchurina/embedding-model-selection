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

We intentionally included ModernBERT from HuggingFace, a base language model that has not been trained for sentence similarity tasks. Here it serves as a negative example, illustrating behaviour of  unsuitable model. By the way, kudos to the model’s creators—it is an impressive achievement.

Both authors work with biomedical data, so we use medical terms and texts for analysis. However, the evaluation methods presented here are universal and can be adapted for any specialized field: whether fintech, legaltech, e-commerce, or other industries.

Let’s dive in!

## Standard Approach to Search Quality Evaluation

Vector search typically follows these steps:

1. An embedding model generates vector representations for documents in a database.
2. A user query is converted into a vector using the same model.
3. The query vector is compared to document vectors.
4. A ranked list of the most relevant documents is returned.

In an ideal scenario, if we have

- a set of documents $D$,
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

This is a great starting point for those who work in healthcare. But if your field is, say, legaltech, BEIR might not be as relevant.

So, while BEIR and MTEB can help shortlist candidate models, the final decision still requires further evaluation.

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

In this pair, synonyms are used:
- "Earwax removal is sometimes necessary when natural cleaning mechanisms fail"
- "When cerumen does not clear on its own, medical intervention may be required"

Here, the disease name is written out in full in one query and abbreviated in the other:
- "What are the major predispositions for renal failure?"
- "Which factors contribute to the development of CKD?"

This pair demonstrates a biological term in its abbreviation and full form:
- "Role of M1dG in oxidative DNA damage"
- "Impact of malondialdehyde-deoxyguanine adducts on genomic stability"

Each pair contains queries with similar meaning, but different pairs represent distinct topics.

Such query pairs can be taken from the service's history, if available, or created with the help of a domain expert. It is important that queries within each pair are semantically close, while different pairs remain distinct. In these examples, they are not fully synonymous from a biologist's perspective but are sufficiently similar.

We converted each query into a vector and calculated pairwise cosine distances between all embeddings. Since queries within the same pair should be close in meaning and queries from different pairs should be distinct, we expect the model to reflect this pattern:
- Lower distances for queries within the same pair,
- Higher distances for queries across different pairs.

The results are visualized in the heatmaps below, where color intensity represents similarity. The diagonal reflects synonymous queries, while off-diagonal values show unrelated queries.

![image](https://github.com/user-attachments/assets/a3819650-557a-4c7e-956f-17fe7ba5ba73)

**OpenAI (text-embedding-3-large):** the diagonal is clearly distinguished against the low similarity scores for queries from different pairs. The model confidently differentiates similar queries within pairs and almost entirely eliminates high scores for unrelated queries.

![image](https://github.com/user-attachments/assets/e1a0762a-0266-436a-8126-1b79a1334271)

**Voyage (voyage-large-2):** the diagonal is poorly defined, indicating that the model struggles to distinguish between similar and dissimilar queries. In several cases, unrelated queries show cosine similarity scores as high as those on the diagonal, suggesting poor separation.

![image](https://github.com/user-attachments/assets/8b10ccd4-770e-46ac-82fb-771f5914b5b0)

**Alibaba (gte-large-en-v1.5):** the diagonal is better defined than in Voyage but has gaps, meaning the model correctly handles some terms but fails with others. The contrast with the background indicates that the model differentiates queries, but not as sharp as OpenAI.

![image](https://github.com/user-attachments/assets/dcf7e604-9edb-4b98-b590-499dbaf4412b)

**Jina (jina-embeddings-v3):** the diagonal is highly pronounced, closely resembling OpenAI's results. However, there are gaps on the diagonal and clusters of high similarity scores off-diagonal, suggesting that the model struggles with this test.

![image](https://github.com/user-attachments/assets/3d4c2689-1e70-443a-a575-439a651d9c2c)

**BioBERT and MedEmbed:** both domain-specific biomedical embeddings show moderate performance. BioBERT provides sharper query separation, while MedEmbed takes a softer approach. However, both models struggle with semantics, as seen from gaps along the diagonal.

![image](https://github.com/user-attachments/assets/cdb512f8-2f2a-484d-97c8-a63871fa6a36)

**ModernBERT-gte (gte-modernbert-base):** performs well in this test. The diagonal is generally distinguishable, with few high scores off-diagonal. Non-diagonal values remain relatively high compared to the diagonal, meaning the model softly separates non-synonymous queries that still belong to the same domain.

![image](https://github.com/user-attachments/assets/eba41827-b587-4051-b471-6cfff800e7d7)

**ModernBERT-base:** A model can behave similar to any of the previously analyzed ones, depending on whether strict or more relaxed query separation is required. However, results should not look like ModernBERT-base: the lack of a visible diagonal suggests an inability to distinguish similar queries within the same domain.

The choice of model depends on the application. If strict query separation is required, OpenAI’s model is a strong candidate. For a more flexible approach, Voyage or ModernBERT-gte may be considered. The key takeaway is that the model should not behave like ModernBERT-base in the given visualization, where all queries appear equally relevant.

### How Query and Document Embeddings Work Together

Query and document embeddings must interact in a way that ensures accurate retrieval. To evaluate this interaction, we will need some metadata, such as category hierarchies, taxonomies, or knowledge graphs.

In biomedical data, we can use MeSH (Medical Subject Headings), a hierarchical system of medical terms. In e-commerce, a similar structure can be derived from product catalogs, in HR-tech, it may come from job title classifications. In other domains, tags, genres, or other classification schemes can serve as taxonomies.

Here's what [MeSH terms](https://meshb.nlm.nih.gov/treeView) look like

![image](https://github.com/user-attachments/assets/04273aa7-7931-4a69-9725-77af9d9110b6)

For this test, we used annotations of biomedical research papers from the last five years, specifically those classified under the MeSH term _Diabetes, Gestational [C19.246.200]_ — a type of diabetes that develops during pregnancy and typically resolves after childbirth.

As a neighboring category, we selected _Latent Autoimmune Diabetes in Adults [C19.246.656]_ (LADA) —  a slow-progressing autoimmune form of diabetes diagnosed in adulthood. For LADA, we only used the category title without associated texts.

Since an average token consists of approximately 3/4 of a word (i.e., 100 tokens ≈ 75 words), we excluded annotations longer than 600 words to avoid chunking, as not all models tested support long text sequences.

So, for analysis, we selected PubMed papers annotations corresponding to two clinically similar diabetes types that differ in etiology.

![image](https://github.com/user-attachments/assets/5bb62867-e717-4616-84d7-197b1cb79288)

The test follows this logic: if two documents belong to the same category, their embeddings should be closer to their own category title than to the title of the neighboring one.

To quantify this, we measure the similarity between documents and category titles. For each document in a category, we compare its embedding with
1. An embedded title of their own category (e.g., "Gestational diabetes").
2. An embedded title of the neighboring category (e.g., "Latent autoimmune diabetes in adults").

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

8. **ModernBERT-base**: The model behaves incorrectly. Documents from _Gestational diabetes_ have a higher average similarity to _LADA_ than to their own category. The average similarity to LADA exceeds 0.8, while to Gestational diabetes it is only ~0.7. This suggests that the model fails to correctly distinguish categories.

Visualizing similarity distributionss helps assess how well a model handles texts in your domain and provides insights into the similarity values typical for each model. It reveals the cosine distance range between documents and queries while also offering intuition about the shape of the distribution. These insights can guide the selection of appropriate threshold values for document filtering in retrieval tasks, helping you balance recall and precision in search applications.

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

![image](https://github.com/user-attachments/assets/8ad1f4c1-c370-4af3-bf0d-e0b9ebe05f03)

To evaluate typo tolerance, we measured the cosine similarity between the embeddings of original medical terms and their misspelled versions across various error types. Based on these results, we generated a heatmap all models.

1. **OpenAI** (text-embedding-3-large) confidently handles most terms, including difficult cases like "therapy → therpay" (0.71), which confused other models. Since from the first test we expect similarity for identical queries to be no lower than ~0.5, we see that "delstrigo → deltrigo" (0.45) and "patient → aptient" (0.43) fall slightly below this threshold. This suggests that while the model does not always correctly recognize typos, it generally assigns a high similarity level to most examples.
    
2. **Voyage** (voyage-large-2) appears to be highly resilient to typos. For example, "delstrigo → deltrigo" and "pregnancy → regnancy" received near-perfect similarity scores of 0.99. From the first test, we expect similarity for identical queries to be no lower than ~0.85 — and all scores here exceed that threshold. This indicates that the model handles typos with high confidence.
    
3. **Alibaba** (gte-large-en-v1.5) seems to struggle with typos. From the first test, we expect similarity for identical queries to be no lower than ~0.75, but here, only "delstrigo → deltrigo" (0.84) exceeds this level, while all other values are significantly lower. The model performs poorly on typos, and if user misspellings are expected in the project, this model may be best excluded from consideration.
    
4. **Jina** (jina-embeddings-v3) handles typos inconsistently. From the first test, we expect similarity for identical queries to be no lower than ~0.5 (though the model showed questionable results there, making it difficult to set a clear threshold). In some cases, it performed exceptionally well: "fatigue → fatiguc" (0.90), "cystinuria → cystiunria" (0.90), "delstrigo → deltrigo" (0.89). In others, it failed completely: "therapy → therpay" (0.37), "pregnancy → regnancy" (0.25). The results don't look stable, and the model does not always handle typos confidently.
    
5. **BioBERT** tends to assign low similarity scores, ~0.5 for relevant texts in the first test. Against this backdrop, the model failed to recognize nearly half of the typo-modified terms, indicating mediocre error-handling quality.

6. **MedEmbed** previously tended to assign high similarity scores to relevant data, with an average of ~0.65 for relevant texts. In this test, only about half of the 12 values exceeded the expected threshold, which also does not indicate high typo tolerance.
   
7. **ModernBERT-gte** (gte-modernbert-base): had an average similarity of around ~0.7 for relevant texts in the first test, and most results here align with that level. It performed exceptionally well on "delstrigo → deltrigo" (0.93), "pregnancy → regnancy" (0.93), and "willebrand disease → xillebrand disease" (0.89). The only clear failure is the pair "therapy → therpay" (0.56), but overall, the model demonstrates great typo robustness.
    
8. **ModernBERT-base**, produced chaotic results in previous tests, making it impossible to seriously evaluate its performance in this test.

This analysis shows how differently models handle user input errors. Understanding typo tolerance is crucial for assessing how a model processes noisy data and how reliably it performs in real-world search scenarios. This highlights the importance of a holistic model selection approach, since a model that excels in one aspect may appear vulnerable in another.

### Handling Domain-Specific Terms

### Evaluating Embeddings for Domain-Specific Terms  

To assess the quality of embeddings in a specialized field, it is crucial to understand how well a model processes domain-specific terminology. If a model fails to recognize context and does not associate terms with their synonyms or related concepts, this can lead to loss of relevant information in retrieval tasks.  

To systematically evaluate how a model handles domain-specific terms, we follow these steps:  

- Curate a set of specialized medical terms, including both common and rare terms.  
- Convert them into embedding vectors using the models under evaluation.  
- Compare the resulting vectors against an external reference dataset (e.g., WordNet) to identify the nearest semantic neighbors and determine whether the model "understands" the terms in our domain.  

This is the list of terms we use for this test:  

- lncRNA — long non-coding RNAs involved in gene regulation.  
- BBB disruption therapy — a method of temporarily disrupting the blood-brain barrier to deliver drugs to brain tissue.  
- Antisense oligonucleotide — short synthetic DNA or RNA strands used in the treatment of genetic diseases and cancer.  
- PD-L1 mAbs — monoclonal antibodies that help the immune system recognize and destroy cancer cells.  
- Kabuki syndrome — a rare genetic disorder.  
- Waldenström Macroglobulinemia — a rare type of non-Hodgkin lymphoma.  
- Frey syndrome — a condition that causes facial sweating while eating.  
- Ozempic — a medication used for Type 2 diabetes treatment and weight management.  
- Cladribine — a drug used for the treatment of multiple sclerosis.  
- Zolgensma — a gene therapy for treating spinal muscular atrophy.  
- ReoPro — a drug used to prevent blood clotting during vascular procedures.  

If a model’s nearest embeddings correspond to synonyms, related concepts, or terms from the same domain, this suggests that the model correctly captures context and meaning. For example, if "RNA" is linked to "genome" or "ribosome," this indicates that the model accurately represents the term. However, if it associates medical terms with random syllables or unrelated words, this signals potential weaknesses in handling specialized terminology.

The table below demonstrates that not all models handle medical terms effectively.

| Term                            | OpenAI                                    | Voyage                                 | Alibaba                                | Jina                                   | BioBERT                                | MedEmbed                              | ModernBERT-GTE                         | ModernBERT-base                            |
|---------------------------------|-------------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| lncRNA                          | nrna, nuclear_rna, informational_rna      | nuclear_rna, mrna, rna                | rna, informational_rna, mrna          | nrna, rna, nrl                        | lunda, livistona, liliales            | rna, nuclear_rna, nrna                | nlrb, nrna, rna                       | mycophage, chalcid, flecainide        |
| BBB disruption therapy          | blood-brain_barrier, thrombolytic_therapy, clot | therapeutic, therapy, therapeutical   | implosion_therapy, disrupt, therapeutical | implosion_therapy, behavior_therapy, disruption | bobsledding, bd, boding              | disruption, bb, bbs                   | ebbtide, bas_relief, implosion_therapy | feedlot, birthwort, bombastically     |
| Antisense oligonucleotide       | nucleoside, antimetabolite, dideoxyinosine | antibody, recombinant, isoantibody    | nucleoside, didanosine, mrna          | non-nucleoside_reverse_transcriptase_inhibitor | ribonucleic_acid, antineoplastic, knock-down | nucleotide, dna, endonuclease         | antipode, noncoding_dna, informational_rna | queenfish, immensurable, antihistamine |
| PD-L1 mAbs                      | cancer_drug, immunotherapeutic, anti-tnf_compound | monoclonal, monoclonal_antibody, antibody | monoclonal_antibody, immunotherapeutic, monoclonal | immunoglobulin_d, monoclonal_antibody, immunoassay | mam, lgb, mamma                     | monoclonal_antibody, monoclonal, pd    | lablab, l-p, blood_profile           | l-plate, hsv-2, 401-k_plan            |
| Kabuki syndrome                 | kakke_disease, noonan's_syndrome, syndrome | syndrome, korsakoff's_syndrome, abasia | kallman's_syndrome, ekbom_syndrome, syndrome | ekbom_syndrome, akinesia, korsakoff's_syndrome | kaki, kalansuwa, kawaka               | korzybski, kaki, ekbom_syndrome       | kuki, ekbom_syndrome, kuki-chin       | giant_reed, cushaw, mortal_sin       |
| Waldenström Macroglobulinemia    | plasmacytoma, myeloma, gammopathy         | agammaglobulinemia, hypogammaglobulinemia, granulomatosis | agammaglobulinemia, multiple_myeloma, myeloma | agammaglobulinemia, myoglobinuria, megaloblast | myoglobinuria, mulishness, hypogammaglobulinemia | agammaglobulinemia, waldenses, hypogammaglobulinemia | wilms_tumour, blood_profile, williams_syndrome | osteosclerosis_congenita, drepanocytic_anaemia |
| Frey syndrome                   | hyperhidrosis, diaphoresis, polyhidrosis  | syndrome, williams_syndrome, idiopathy | frey, bruxism, waterhouse-friderichsen_syndrome | syndrome, frey, reye's_syndrome       | frey, freyr, freya                     | frey, freya, freyr                     | reye's_syndrome, frey, ramsay_hunt_syndrome | frostwort, foumart, fortunella       |
| Ozempic                         | glucophage, glipizide, zapotecan         | otic, zocor, empirin                   | ozena, obidoxime_chloride, oxyphencyclimine | ocimum, omotic, oecumenic              | ozena, ozarks, ozaena                   | ozonize, typic, ozonide                | zyloprim, oxytocic_drug, oxytocic      | palometa, pseudemys, empyreal        |
| Cladribine                      | chlorambucil, leukeran, mercaptopurine    | clonidine, chlorpromazine, clomipramine | zalcitabine, lamivudine, deoxyadenosine | cladrastis, clerid, clinid             | cladonia, cladrastis, cladode          | clad, cladode, zalcitabine             | cuprimine, calcimine, cadaverine       | cladoniaceae, cladophyll, cicadellidae |
| Zolgensma                        | zoloft, lofortyx, vincristine            | zetland, zinzendorf, nijmegen          | zeugma, zygnema, genus_zygnema         | zola, zymogen, zolaesque               | ziziphus, zola, zetland                 | zeugma, glechoma, zygoma               | zeugma, z, malosma                     | schmaltz, spotweld, zinfandel         |
| ReoPro                           | lipo-hepin, thrombolytic_agent, plavix   | pro, re, ream                          | appro, reechoing, recopy               | pro, recco, ro                         | reproval, retem, reamer                 | requital, reenact, reproof             | repp, revisal, reseau                  | galactocele, rectocele, proviso       |

1. **OpenAI (text-embedding-3-large)** performed exceptionally well. For instance, it was the only model that correctly linked term "BBB disruption therapy" to "blood-brain barrier" in WordNet. It also properly associated "PD-L1 mAbs" with cancer drug, capturing its relevance to oncology treatments. For "Frey syndrome", it identified "hyperhidrosis", which matches the syndrome’s clinical manifestation. It also correctly clustered medications related to "Ozempic", grouping it with other type 2 diabetes treatments. However, the model struggled with "Zolgensma" and "ReoPro", though it still retained a general medical context in its associations.

2. **Voyage (voyage-large-2)** delivered moderate performance. For "BBB disruption therapy", it latched onto the word "therapy" but failed to extract its relevant meaning. For "Frey syndrome", it retrieved generic terms containing “syndrome” without capturing its specificity. While "PD-L1 mAbs" was correctly identified as an abbreviation, the model did not link it to its mechanism of action or therapeutic use. "Ozempic" and "ReoPro" were poorly handled, producing token-based outputs such as "otic", "zocor", and "ream". Overall, Voyage provides surface-level processing of medical terms without deep semantic understanding.

3. **Alibaba (gte-large-en-v1.5)** showed inconsistent performance. For instance, it associated "BBB disruption therapy" with psychotherapy-related terms (like "implosion therapy"), which is incorrect. However, it correctly recognized "Antisense oligonucleotide" by identifying molecular components and linked "Waldenström Macroglobulinemia" to tumor-related terms. That said, "Ozempic" failed to yield meaningful associations, and "Zolgensma" remained unrecognized, with nearest terms including "zeugma" and "zygnema", which are unrelated to medicine.

4. **Jina Jina (jina-embeddings-v3)** clearly lacks a medical domain focus. It produced nonsensical syllabic outputs for terms like "lncRNA", "Ozempic", "Cladribine", "Zolgensma", and "ReoPro" instead of meaningful associations. For "BBB disruption therapy", it incorrectly retrieved psychotherapy-related terms ("implosion therapy"). However, it performed better on terms like "Antisense oligonucleotide", "PD-L1 mAbs", and "Waldenström Macroglobulinemia", though it still significantly underperformed compared to OpenAI and Voyage.

5. **BioBERT**, despite being a biomedical model, performed poorly. It failed to correctly process most terms, including "BBB disruption therapy", where it returned "bobsledding". The model displayed a strong bias towards token-based retrieval, failing to generalize term meanings. For "Frey syndrome", it returned words like "frey", "freyr", and "freya", which have no semantic relevance, indicating weak medical domain alignment.

6. **MedEmbed** performed only slightly better than BioBERT. It maintained a general connection to the medical field, making it somewhat usable in this context. However, it failed to handle pharmaceutical terms such as "Ozempic" and "Cladribine:, producing inadequate associations.

7. **ModernBERT-gte (gte-modernbert-base)** showed moderate performance. The model correctly processed "Antisense oligonucleotide," suggesting "antipode" and "noncoding_dna," which partially reflects the meaning of the term. For "Kabuki syndrome," like other models, it latched onto the word "syndrome" but also pulled in "kuki-chin," which is unrelated. Its results for "Waldenström Macroglobulinemia" are decent, since the model identified relatively relevant terms such as "wilms_tumour" and "blood_profile." It failed on "Ozempic," but unlike Jina and BioBERT, at least some of its nearest terms were medical. Overall, the model demonstrates decent but unremarkable performance in handling medical terminology.

8. **ModernBERT-base** is entirely unreliable for medical terminology. More than half of its associations fell outside the medical domain. Instead of relevant terms, the model returned words like "bombastically", "queenfish", "frostwort", and "schmaltz". This indicates that the model is not suitable for handling medical context.

The analysis of domain-specific terms shows that model performance can vary significantly across specialized fields. This test provides a way to **look under the hood** of an embedding model and assess whether it truly captures context for terms relevant to your domain.  

An ideal model should correctly associate terms with their synonyms and related concepts, ensuring high-quality retrieval. However, even the best models have limitations: new or rare terms that were not part of the training data will remain inaccessible to them.

## Bonus Track: Outlier Detection

If you've made it this far, here's a bonus track!

What does your model consider the "opposite" of your query? If you already have a working search system, try retrieving the least relevant documents based on cosine similarity instead of the top-n results.

This technique helps identify what the model deems minimally relevant, which can reveal outliers, documents that significantly differ from the rest. Such data points may require further investigation or additional preprocessing to ensure they belong in your index.

Using this method, we accidentally discovered the following documents in a medical text database:  

- An IKEA wardrobe assembly manual,  
- A list of the best picnic spots in London,  
- A guide to fishing for trout with artificial lures.

This technique serves as a sanity check for large-scale datasets, providing insights into model behavior while helping improve data quality. And honestly? It’s just fun!

## Conclusion

The presented embedding model analysis methods help gain a deeper understanding of their strengths and weaknesses. They enable an informed choice of a model for vector search, even with limited resources, reducing potential risks.  

These tests are qualitative, not quantitative. They do not provide strict numerical evaluations but are cost-effective, require minimal computation, and are understandable even for non-technical stakeholders. The advantage of a qualitative approach is that it improves explainability for the business: it makes it easier to demonstrate how the model behaves, where its limitations lie, and why search may not always work as expected.  

The evaluation methods discussed in this article allow you to build an initial RAG system even without a golden dataset. They provide fast results, helping to eliminate unsuitable candidate models early and solving the cold start problem, allowing you to test a model’s adequacy for your domain before blindly committing to it.  

These tests also reinforce a well-known reality: there is no such thing as a perfect embedding model. Even the best models will not deliver flawless results because:  
- They may not have encountered rare data during training,  
- They have never "seen" new data that emerged after their training.  

Embeddings are not a magic bullet. If your project operates in a specialized or rapidly evolving field, vanilla vector search alone won’t be enough.  

To build a reliable search system, it’s essential to consider hybrid approaches, integrate reranking, or leverage additional knowledge sources. This approach ensures that your search system does more than just return documents, but actually answers user queries.

Thank you for reading the article to the end! If you have any questions or ideas, we'd love to discuss them in the comments. And kudos to [Konstantin Shevchenko](https://www.linkedin.com/in/konstantinshevchenko/) for his help with examples and expert review of the biological tests!

Check out our [GitHub](https://github.com/chakchurina/embedding-model-selection/): where we’ve shared a notebook that you can adapt to your own use cases. We appreciate stars, pull requests, and feedback!

If your business or team needs expertise in text processing, reach out to [Maria](https://www.linkedin.com/in/maria-chakchurina/) or [Ekaterina](https://www.linkedin.com/in/ekaterina-antonova-51108a67/) on LinkedIn — we’d be happy to discuss your needs and help with solutions. Also, follow us so you don’t miss future articles!

## Inspired by

- https://haystackconf.com/eu2023/talk-13/
- https://haystackconf.com/eu2023/talk-7/ 
