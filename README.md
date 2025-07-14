# Yu-Gi-Oh! Set Revenue Analysis

This repository investigates which features of Yu-Gi-Oh! trading card sets are most closely related to their revenue. Using data on card attributes, product types, and external events, we apply machine learning to explore and quantify these relationships.

## Objective

To identify and rank the most influential factors contributing to set-level revenue, using interpretable and reproducible data analysis.

## Approach

We trained a Random Forest regression model using a merged dataset constructed from:

- **Meta Score**: Ordinal encoding of how competitively relevant cards in a set are.
- **Rarity Score**: Average card rarity in a set (e.g., Common = 1 to Secret Rare = 5).
- **Reprints**: Proportion of cards in the set that are reprints.
- **Number of Cards**: Count of unique cards in the set.
- **Product Category**: Categorical encoding of the type of product (e.g., booster, deck).
- **Impacting Events**: Categorical flag indicating if a tournament or banlist occurred within 30 days of release.

## Key Findings

- The most influential features for predicting revenue were:
  1. Product Category
  2. Rarity Score
  3. Meta Score
  4. Reprints
  5. Number of Cards
  6. Impacting Events

- Sets that are more playable out-of-the-box (e.g., starter or structure decks) and include high-rarity or meta-relevant cards tend to have higher revenue.

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib or seaborn
- jupyter

