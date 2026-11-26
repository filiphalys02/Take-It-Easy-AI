# Take It Easy AI — sieć neuronowa w przeglądarce

Repozytorium ma dwie niezależne części:

| Katalog | Do czego służy | Potrzebne do wdrożenia? |
|---|---|---|
| [`app/`](app) | **Aplikacja** — gra „Ty kontra sieć” w przeglądarce (HTML + JS + model ONNX). Cała gra i sieć działają w przeglądarce, bez serwera. | **tak** |
| [`training/`](training) | **Trening modelu** — silnik zasad, sieć, skrypt treningowy, eksport do ONNX, testy. | nie |
| [`.github/workflows/pages.yml`](.github/workflows/pages.yml) | Publikuje `app/` na GitHub Pages. | **tak** |

Przepływ: `training/` uczy sieć → `training/model.pt` → `export_onnx.py` → `app/model.onnx` → gra w przeglądarce.

```
V2/
├── app/                      ← APLIKACJA (to trafia na GitHub Pages)
│   ├── index.html            interfejs (EN/PL, desktop i telefon)
│   ├── engine.js             zasady i punktacja (port tie_env.py)
│   ├── ai.js                 expectimax nad siecią (port action_values)
│   ├── ai-worker.js          Web Worker: ONNX Runtime Web + ai.js
│   ├── model.onnx            sieć (generowana przez training/export_onnx.py)
│   └── vendor/               ONNX Runtime Web (MIT), bez CDN
├── training/                 ← TRENING (nie jest publikowany)
│   ├── tie_env.py            zasady gry, plansza, punktacja
│   ├── tie_value.py          sieć HexLineNet, expectimax, samogranie
│   ├── train_value.py        skrypt treningowy
│   ├── export_onnx.py        model.pt → app/model.onnx
│   ├── test_rules.py         testy zasad i punktacji
│   ├── requirements.txt
│   └── model.pt              wytrenowany model (PyTorch)
└── .github/workflows/pages.yml
```

---

## Szybki start

**Uruchomienie aplikacji lokalnie** (moduły ES i workery nie działają z `file://`, więc potrzebny jest serwer):

```bash
python -m http.server 8000 -d app       # http://127.0.0.1:8000/
```

**Trening i eksport modelu** (z katalogu `training/`):

```bash
cd training
pip install -r requirements.txt
pip install onnx onnxscript onnxruntime   # tylko do eksportu

python train_value.py                     # trening -> model.pt (opcjonalny, gotowy model.pt jest w repo)
python export_onnx.py                     # model.pt -> ../app/model.onnx
python test_rules.py                      # testy zasad
```

Wymagania: Python 3.10+, `numpy`, `torch` ≥ 2.0 (GPU zalecane do treningu). Do samej aplikacji Python nie jest potrzebny (poza serwerem lokalnym do testów).

---
## Zasady gry (jak są zaimplementowane)

* 27 kafelków = {1, 5, 9} × {2, 6, 7} × {3, 4, 8}. Liczba z {1,5,9} biegnie **pionowo**, z {2,6,7} wzdłuż przekątnej **„/”**, z {3,4,8} wzdłuż przekątnej **„\\”**.
* Plansza: 19 pól w kolumnach 3‑4‑5‑4‑3. W partii pada 19 z 27 kafelków, ten sam kafelek dostają obaj gracze.
* Punkty: każda z 15 linii (5 pionowych, 5 „/”, 5 „\\”; długości 3, 4, 5, 4, 3) daje *(wspólna liczba) × (długość linii)*, jeśli wszystkie kafelki w linii mają tę samą liczbę w kierunku tej linii.

Numeracja pól (w kodzie 0–18, dla człowieka 1–19), kolumnami od lewej, w kolumnie od góry:

```
 kol0  kol1  kol2  kol3  kol4
  1     4     8    13    17
  2     5     9    14    18
  3     6    10    15    19
        7    11    16
             12
```

**Orientacja przekątnych.** Domyślnie 2/6/7 biegną po „/”, a 3/4/8 po „\\” (zgodnie z opisem w repozytorium polarbart/TakeItEasyAI). Jeśli Twoja fizyczna plansza ma to odwrotnie, ustaw `SWAP_DIAGONALS = True` w `training/tie_env.py` (oraz `AXES` w `app/engine.js`) i **wytrenuj model od nowa** — sieć uczy się pod konkretną orientację.

---

## Trening

```bash
cd training
python train_value.py                      # ustawienia domyślne (3000 iteracji), wynik: model.pt
```

### Jak to działa

Gra jest jednoosobowa, a jej model jest w pełni znany (położenie kafelka jest deterministyczne, kolejny kafelek jest losowany jednostajnie spośród niewidzianych). Zamiast uczyć politykę „na ślepo”, sieć uczy się funkcji wartości planszy **po położeniu kafelka** (afterstate):

```
W(plansza) = E_kafelek [ max_pole ( punkty_zamknięte_teraz + W(plansza po ruchu) ) ]
```

Zbiór niewidzianych kafelków wynika z planszy, więc wystarczy sieć `W(plansza)`. Pętla treningowa:

1. **Samogranie** aktualną polityką (zachłanną po `W`, z losową eksploracją) → bufor plansz osobno dla każdej liczby położonych kafelków.
2. **Cele**: dla próbki plansz liczone jest dokładne oczekiwanie po wszystkich niewidzianych kafelkach (maksimum po polach) z użyciem wolno aktualizowanej sieci docelowej — bez wariancji losowania.
3. **Regresja** `W(plansza) → cel` z augmentacją obrotem planszy o 180° (jedyna symetria punktacji) i miękką aktualizacją sieci docelowej (`--tau`).

Sieć `HexLineNet` przekazuje komunikaty między polami a 15 liniami planszy (zna strukturę gry), a wynik to suma potencjałów linii.

### Ważniejsze opcje

| Opcja | Domyślnie | Znaczenie |
|---|---|---|
| `--iterations` | 3000 | liczba iteracji |
| `--games-per-iter` | 512 | partii samogrania na iterację |
| `--states-per-k` | 512 | plansz na każdą liczbę kafelków w iteracji (koszt liczenia celów rośnie liniowo) |
| `--target-depth` | 1 | `2` = cele z 2‑krokowego expectimaxu (destylacja przeszukiwania do sieci) |
| `--target-tiles` | 8 | próbkowane kafelki na górnych poziomach (przy `--target-depth ≥ 2`) |
| `--d`, `--blocks` | 128, 4 | szerokość i liczba bloków sieci |
| `--lr`, `--tau` | 1e‑3, 0.01 | krok uczenia (cosinus do 1e‑5), tempo sieci docelowej |
| `--eps-max` | 0.4 | maks. losowość eksploracji (maleje w trakcie treningu) |
| `--eval-every`, `--eval-games` | 50, 2000 | ewaluacja zachłannej polityki na stałych partiach |
| `--out` | `model.pt` | plik wyjściowy (zapisywany przy najlepszym wyniku ewaluacji) |
| `--device` | auto | `cuda` / `cpu` |

Przykład dla mocnego GPU:

```bash
python train_value.py --iterations 8000 --target-depth 2 --d 256 --blocks 6 \
                      --games-per-iter 2048 --states-per-k 2048
```

### Co pokazuje log

* co 10 iteracji: **RMSE celu** (w punktach), **W(pusta plansza)** (szacunek oczekiwanego wyniku pod optymalną grą), średni wynik samogrania, `eps`, `lr`;
* co `--eval-every` iteracji: średnia ± błąd standardowy, odchylenie, percentyle p10/p50/p90 i P(wynik ≥ 100/150/200) na **stałym** zestawie partii (te same losowania w każdej ewaluacji → porównania sparowane);
* na końcu: 500 tych samych partii dla sieci zachłannej i dla sieci z expectimaxem (głębokość 2 + dokładna końcówka) wraz ze **średnim zyskiem z przeszukiwania**.

Jak czytać metryki: celem jest **oczekiwany wynik końcowy**. `W(pusta plansza)` powinno rosnąć i zbliżać się do wyniku ewaluacji; jeśli wyraźnie go przekracza, wartości są zawyżone (zmniejsz `--tau`). Duży zysk z przeszukiwania oznacza, że sieć jeszcze nie jest zbieżna albo jest za mała. Przy odchyleniu rzędu 25–30 pkt na partię potrzeba ≥ 2000 partii, żeby błąd standardowy wynosił ok. 0,6 pkt.

---

## Aplikacja w przeglądarce (`app/`)

| Plik | Rola |
|---|---|
| `app/index.html` | Interfejs (HTML + CSS + JS, rysowanie w SVG): wybór języka EN/PL (domyślnie EN), układ desktopowy i mobilny. |
| `app/engine.js` | Port `tie_env.py`: zasady, punktacja, partia Ty vs sieć. |
| `app/ai.js` | Expectimax d=2 nad siecią (port `action_values`), dokładne rozwiązanie przy ≤ 5 wolnych polach. |
| `app/ai-worker.js` | Web Worker: ONNX Runtime Web (WASM) + `ai.js`, żeby nie blokować interfejsu. |
| `app/model.onnx` | Sieć wyeksportowana z `model.pt`. |
| `app/vendor/` | ONNX Runtime Web 1.31.1 (licencja MIT) — pliki zapisane w repo, bez CDN. |

### Jak wygląda rozgrywka

* **Góra:** wylosowany kafelek, numer tury (1–19) i przełącznik języka (EN / PL).
* **Plansze:** Twoja i sieci, z bieżącym wynikiem i rozbiciem ukończonych linii. Zawsze ruszasz pierwszy, potem sieć kładzie ten sam kafelek na swojej planszy.
* **Komputer:** po najechaniu na wolne pole widać podgląd kafelka, kliknięcie kładzie go.
* **Telefon:** układ w jednej kolumnie; dotknięcie wolnego pola pokazuje podgląd, ponowne dotknięcie kładzie kafelek (ochrona przed przypadkowym ruchem).
* **Dół:** kafelki, które zostały w talii.
* **Koniec gry:** okno z wynikiem i przyciskiem *PLAY AGAIN* (działa też Enter / Spacja).

### Siła przeciwnika

Sieć gra expectimaxem głębokości 2 (jak `action_values(depth=2, exact_last=5)` w `tie_value.py`), ale na głębokości 2 sprawdzanych jest tylko 5 najlepszych ruchów wg sieci i obliczenia nie są uśredniane z obrotem planszy. W teście 100 partii daje to tę samą średnią co pełny expectimax (165,2 pkt; różnica sparowana 0,09 ± 0,09), a ruch zajmuje ok. 1,5 s zamiast ok. 10 s. Ustawienia są w `app/ai-worker.js` (`OPTS`).

### Eksport modelu

Po każdej zmianie `training/model.pt` (z korzenia repozytorium):

```bash
python training/export_onnx.py     # model.pt -> app/model.onnx (+ kontrola zgodności z PyTorch)
```

### Test lokalny

Moduły ES i workery nie działają z `file://`, więc użyj serwera:

```bash
python -m http.server 8000 -d app     # http://127.0.0.1:8000/
```

### Wdrożenie na GitHub Pages

1. Wypchnij repozytorium na GitHuba (`V2` jako korzeń repozytorium).
2. *Settings → Pages → Build and deployment → Source:* **GitHub Actions**. Workflow `.github/workflows/pages.yml` publikuje `app/` po każdym pushu na `main`.
3. Strona: `https://<użytkownik>.github.io/<repozytorium>/`.

Uwagi: Pages z prywatnego repozytorium wymaga płatnego planu GitHuba, a opublikowana strona jest publiczna. Jeśli `V2` nie jest korzeniem repozytorium, przenieś `.github/` do korzenia i zmień `path: app` na `path: V2/app` w workflow.

---

## Testy

```bash
python training/test_rules.py        # albo: python -m pytest training/test_rules.py
```

Testy nie wymagają torcha. Sprawdzają strukturę planszy (15 linii), zgodność punktacji z niezależną, wolną implementacją i zasady `GameState`.

---

## Rozwiązywanie problemów

| Problem | Rozwiązanie |
|---|---|
| Biała strona / błąd w konsoli po otwarciu `app/index.html` z dysku | Użyj serwera: `python -m http.server 8000 -d app`. |
| Komunikat „The AI could not run” | Sprawdź, czy `app/model.onnx` i `app/vendor/*` są w repozytorium i zostały opublikowane. |
| Rada sieci nie pasuje do fizycznej planszy | Sprawdź orientację przekątnych i ewentualnie ustaw `SWAP_DIAGONALS` (patrz wyżej), potem wytrenuj model od nowa. |
| `W(pusta plansza)` wyraźnie większe niż wynik ewaluacji | Zawyżone wartości: zmniejsz `--tau`, wydłuż trening, rozważ `--target-depth 2`. |
| Brak pamięci GPU przy treningu | Zmniejsz `--states-per-k`, `--games-per-iter` lub rozmiar sieci (`--d`, `--blocks`). |
