# Randstad & MKERS Sentiment Analysis Challenge

[![Verifica su dati sintetici](https://github.com/BartoloMorgigno11/1st-Place-AI-Hackathon---NLP-Sentiment-Analysis/actions/workflows/portfolio-ci.yml/badge.svg)](https://github.com/BartoloMorgigno11/1st-Place-AI-Hackathon---NLP-Sentiment-Analysis/actions/workflows/portfolio-ci.yml)

[English version](README.md)

## Panoramica
Questo repository documenta una soluzione di sentiment analysis realizzata nell'ambito dell'AI Hackathon Randstad e MKERS al Politecnico di Bari. Il [post pubblico degli organizzatori, condiviso da un componente della squadra](https://it.linkedin.com/posts/bartolo-morgigno_grazie-mille-per-la-bellissima-esperienza-activity-7262877376995328000-8j8m), conferma la vittoria del team. Il codice impiega TF-IDF e regressione logistica bilanciata; i dataset e i report numerici originali non sono pubblicati, quindi l'accuratezza ottenuta durante il contest non è riproducibile indipendentemente da questo repository.

Oltre al modello, forniamo anche una visualizzazione geografica dei punteggi medi dei recensori in Europa.

**Team vincitore:** Bartolo Morgigno, Erica Porcelli, Vito Marzocca e Gabriele Colapinto. Il lavoro di sviluppo è stato condiviso equamente, senza una suddivisione formale dei componenti.

## Punti di Forza
- Hackathon: membro del team vincitore, come documentato dal [post degli organizzatori](https://it.linkedin.com/posts/bartolo-morgigno_grazie-mille-per-la-bellissima-esperienza-activity-7262877376995328000-8j8m).
- Semplice ma potente: TF-IDF su unigrammi e bigrammi con logistic regression bilanciata.
- Insight immediati: geomap europea con i punteggi medi aggregati per nazione.
- Script riproducibili: tutto eseguibile da terminale senza usare i notebook.

## Struttura del Repository
```
Geomap/
  geomap.ipynb
Reviews/
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
3. Procurati dati che sei autorizzato a usare. I CSV originali dell'hackathon non sono distribuiti in questa edizione pubblica per portfolio. Vedi [Configurazione dei dati](#configurazione-dei-dati).
4. Allena e valuta il modello di sentiment:
   ```bash
   python scripts/train_sentiment_model.py --train-path path/del/training.csv
   ```
   Alla prima esecuzione lo script scarica automaticamente le stopwords NLTK. I risultati vengono stampati a video e la curva ROC insieme alla confusion matrix sono salvate nella cartella `outputs/`.
5. Genera la geomap europea:
   ```bash
   python scripts/generate_geomap.py --data-path path/del/punteggi_per_citta.csv
   ```
   L'immagine risultante viene salvata in `outputs/european_sentiment_map.png`. Aggiungi `--show` per visualizzarla a schermo.

## Verifica riproducibile con dati sintetici

I CSV originali dell'hackathon non sono distribuiti in questo repository. Per consentire una prova
indipendente, il repository genera **16 recensioni artificiali** con etichette
binarie bilanciate. Il risultato ottenuto su questi dati serve esclusivamente a
verificare che gli script funzionino: **non riproduce i risultati del contest, non misura
la generalizzazione del modello e non contiene recensioni reali**.

```bash
python -m pip install -r requirements.txt
python scripts/generate_synthetic_fixture.py
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/train_sentiment_model.py --train-path examples/synthetic_reviews.csv --skip-extra
```

Il training scarica le stopword inglesi di NLTK al primo avvio, se mancanti.
Gli output dimostrativi sono salvati in `outputs/`. I test unitari possono
essere eseguiti senza dataset originali e senza rete; GitHub Actions esegue
anche un test completo della CLI sui dati artificiali.

## Personalizzazioni
- Usa un dataset differente con `--train-path path/del/dataset.csv`.
- Valuta un dataset aggiuntivo con etichette tramite `--extra-path path/del/dataset_extra.csv`.
- Modifica la dimensionalità del TF-IDF con `--max-features`.
- Cambia la scala cromatica della geomap con `--vmin` e `--vmax`.

Esegui `python scripts/train_sentiment_model.py --help` o `python scripts/generate_geomap.py --help` per un elenco completo delle opzioni disponibili.

## Configurazione dei dati

I dataset del contest non sono inclusi. Non aggiungerli a fork o copie pubbliche senza un permesso esplicito di redistribuzione dal proprietario.

- Il CSV di training deve includere `Review` (testo libero) e `Promotore` (etichetta binaria). Passalo con `--train-path`; un set aggiuntivo etichettato può essere fornito con `--extra-path`.
- Il CSV per la geomap deve includere `City_Nation` (ad esempio `Bari, Italy`) e `Reviewer_Score` numerico. Passalo con `--data-path`.
- Lo shapefile in `map/` proviene da Natural Earth (public domain).

## Output Generati
- `outputs/roc_curve.png` - curva ROC sul validation set.
- `outputs/confusion_matrix.png` - confusion matrix 2x2 con etichette.
- `outputs/european_sentiment_map.png` - mappa termica dei punteggi medi per nazione.

## Note sul Versionamento
- `.gitignore` mantiene fuori da Git gli artefatti generati, gli ambienti virtuali, la cartella `outputs/` e i CSV del contest.

## Licenza
I dataset e gli artefatti derivati appartengono agli organizzatori dell'hackathon e al team partecipante. Gli shapefile Natural Earth sono di pubblico dominio. Salvo diversa indicazione, il codice presente in questo repository è distribuito con licenza MIT.
