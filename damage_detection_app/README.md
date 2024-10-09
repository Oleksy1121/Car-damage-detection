# YOLO Object Detection Image Viewer

## Opis
Aplikacja służy do wykrywania uszkodzeń karoserii samochodowej przy użyciu modelu YOLO (You Only Look Once). Umożliwia ładowanie folderu ze zdjęciami, detekcję uszkodzeń oraz wyświetlanie wyników w postaci naniesionych adnotacji na obrazach. Aplikacja posiada graficzny interfejs użytkownika (GUI) oparty na bibliotece Tkinter, który umożliwia łatwą nawigację po zdjęciach oraz przeglądanie wyników detekcji.

## Funkcje
- Ładowanie obrazów z wybranego folderu.
- Detekcja uszkodzeń karoserii samochodowej za pomocą modelu YOLO.
- Wyświetlanie obrazów z oznaczonymi obszarami wykrytych uszkodzeń.
- Nawigacja po obrazach za pomocą przycisków "Poprzedni" i "Następny".
- Log prezentujący szczegóły dotyczące wykrytych uszkodzeń.
- Obsługa przeciągania i upuszczania folderów.
- Automatyczne dopasowanie wyświetlanego obrazu do rozmiaru okna aplikacji.

## Przykładowe zdjęcie aplikacji
Poniżej znajduje się przykładowe zdjęcie przedstawiające działanie aplikacji:

![Przykład aplikacji](img/app_screen.png)

## Uruchomienie aplikacji
Zaleca się otwarcie folderu z aplikacją w środowisku programistycznym PyCharm jako projekt. W folderze projektu znajduje się wirtualne środowisko `venv`, które zawiera wszystkie niezbędne zależności.

1. Otwórz projekt w PyCharm, wskazując folder główny aplikacji.
2. Aktywuj wirtualne środowisko (`venv`), które jest zintegrowane z projektem.
3. Zainstaluj wymagane pakiety, uruchamiając poniższe polecenie w terminalu:
   ```bash
   pip install -r requirements.txt
   ```
4. Uruchom aplikację za pomocą pliku app.py
      ```bash
   python app.py
   ```
