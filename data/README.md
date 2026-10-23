# Dataset

This project uses **Bend The Truth**, an Urdu fake news dataset by Amjad et al. (2020), published at
https://github.com/MaazAmjad/Datasets-for-Urdu-news.

- 900 news articles: 500 real and 400 fake.
- Five domains: business, health, entertainment (showbiz), sports and technology.
- Official split: 638 training and 262 test articles.
- Each fake article is a rewrite of a real article by a professional journalist. Pairs share a file name, e.g. `bus12.txt` in both `Real/` and `Fake/`.

The dataset isn't stored in this repository. Download it with:

```bash
python -c "from src.urdu_text import download_corpus; download_corpus()"
```

This saves it to `data/raw/1.Corpus/{Train,Test}/{Real,Fake}/*.txt`. The demo app downloads it automatically the first time it needs sample articles.

## Citation

> Maaz Amjad, Grigori Sidorov, Alisa Zhila, Helena Gómez-Adorno, Ilia Voronkov and Alexander Gelbukh.
> *Bend the Truth: A Benchmark Dataset for Fake News Detection in Urdu and Its Evaluation.*
> Journal of Intelligent & Fuzzy Systems, 39(2), 2457–2469, 2020. https://doi.org/10.3233/JIFS-179905
