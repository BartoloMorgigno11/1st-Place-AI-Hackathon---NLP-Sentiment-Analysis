# Randstad & MKERS Sentiment Analysis Challenge

## Panoramica
Questo repository contiene la soluzione con cui abbiamo vinto l'AI Hackathon organizzato da Randstad e MKERS al Politecnico di Bari. L'obiettivo era costruire nel minor tempo possibile un sistema di sentiment analysis su recensioni di hotel ottenendo la massima accuracy. Grazie a un pipeline di preprocessamento mirato e a un classificatore di regressione logistica ben bilanciato abbiamo raggiunto il 96 percento di accuracy e conquistato il primo posto.

Oltre al modello, forniamo anche una visualizzazione geografica dei punteggi medi dei recensori in Europa.

## Punti di Forza
- Vincitori dell'hackathon: accuracy del 96 percento in tempi rapidissimi.
- Semplice ma potente: TF-IDF su unigrammi e bigrammi con logistic regression bilanciata.
- Insight immediati: geomap europea con i punteggi medi aggregati per nazione.
- Script riproducibili: tutto eseguibile da terminale senza usare i notebook.

## Struttura del Repository
```
Geomap/
  data_geomap_total.csv
  geomap.ipynb
Reviews/
  data_classification_20.csv
  data_classification_80.csv
  logistic_regression.ipynb
map/
  ne_110m_admin_0_countries.*  (shapefile Natural Earth)
scripts/
  generate_geomap.py
  train_sentiment_model.py
requirements.txt
README.md
README_ITA.md
```

## Avvio Rapido
1. Crea un ambiente virtuale (consigliato):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
3. Allena e valuta il modello di sentiment:
   ```bash
   python scripts/train_sentiment_model.py
   ```
   Alla prima esecuzione lo script scarica automaticamente le risorse NLTK necessarie (stopwords e punkt). I risultati vengono stampati a video e la curva ROC insieme alla confusion matrix sono salvate nella cartella `outputs/`.
4. Genera la geomap europea:
   ```bash
   python scripts/generate_geomap.py
   ```
   L'immagine risultante viene salvata in `outputs/european_sentiment_map.png`. Aggiungi `--show` per visualizzarla a schermo.

## Personalizzazioni
- Usa un dataset differente con `--train-path path/del/dataset.csv`.
- Valuta un dataset aggiuntivo con etichette tramite `--extra-path path/del/dataset_extra.csv`.
- Modifica la dimensionalita del TF-IDF con `--max-features`.
- Cambia la scala cromatica della geomap con `--vmin` e `--vmax`.

Esegui `python scripts/train_sentiment_model.py --help` o `python scripts/generate_geomap.py --help` per un elenco completo delle opzioni disponibili.

## Note sui Dati
- I dataset delle recensioni sono in `Reviews/` e includono la colonna `Review` (testo) e `Promotore` (etichetta binaria).
- `Geomap/data_geomap_total.csv` contiene punteggi aggregati per citta; lo script per la mappa li aggrega ulteriormente per nazione.
- Lo shapefile in `map/` proviene da Natural Earth (public domain).

## Output Generati
- `outputs/roc_curve.png` - curva ROC sul validation set.
- `outputs/confusion_matrix.png` - confusion matrix 2x2 con etichette.
- `outputs/european_sentiment_map.png` - mappa termica dei punteggi medi per nazione.

## Note sul Versionamento
- `.gitignore` mantiene fuori da Git gli artefatti generati, gli ambienti virtuali e la cartella `outputs/`.
- Conserva o rimuovi i dataset dal controllo versione in base alle policy di condivisione stabilite dagli organizzatori dell'hackathon.

## Licenza
I dataset e gli artefatti derivati appartengono agli organizzatori dell'hackathon e al team partecipante. Gli shapefile Natural Earth sono di pubblico dominio. Salvo diversa indicazione, il codice presente in questo repository e' distribuito con licenza MIT.
