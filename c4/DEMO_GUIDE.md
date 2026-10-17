# How to run the C4 demo, offline, step by step

Everything C4 needs is already on this laptop. No internet is required.

## A. Turn the internet off

1. Click the **Wi-Fi icon** at the bottom-right of the screen.
2. Click **Airplane mode** (it turns blue). If a network cable is plugged in,
   unplug it.

## B. Open a terminal

1. Open **VS Code**.
2. In the top menu, click **Terminal → New Terminal**. A panel opens at the
   bottom.

## C. Go to the C4 folder

Copy this line, paste it into the terminal and press **Enter**:

    cd C:\Users\osadi\source\repos\J26-DS-313-SORA\c4

## D. Start the demo

Paste this and press **Enter**:

    python src/demo.py

Wait. It first prints `loading models (about a minute) ...`. After 30 to 60
seconds it prints:

    C4 demo running offline at http://127.0.0.1:8765  (Ctrl+C to stop)

**Do not close this terminal.** The demo runs only while it stays open.

## E. Open the demo in the browser

1. Open **Chrome** or **Edge**.
2. Click the address bar at the top and type **exactly**:

       http://127.0.0.1:8765

3. Press **Enter**. The C4 page opens.

`127.0.0.1` means "this laptop". The browser is talking to the demo running on
your own computer, not to the internet, which is why it works with Airplane
mode on.

## F. Use it

The page has one text box and one button. Only what you type is processed; no
stored data is ever loaded.

1. Type or paste any sentence into the big box.
2. Click **Redact this text**.
3. Wait for **"Done."**. The results appear below:
   - the table of what was found
   - the original text with highlights
   - the redacted text

Sentences to try:

    My name is Tharindu Fernando. මගේ නම තරිඳු ප්‍රනාන්දු.

→ both names become the same [PERSON_1], shown as "Latin + Sinhala"
(cross-script linking, Contribution 2).

    mage NIC eka 9 5 3 2 0 1 4 5 6 v, number eka binduwai hatai hatai ekai dekai thunai hatharai pahai hayai hatha.

→ the NIC and the phone number spoken as words are both redacted.

    Seylan Bank eken call kala. Mama rupiyal 25000 gewwa.

→ nothing is redacted: a bank name and a money amount are not personal data.

## G. Stop the demo

Click inside the terminal and press **Ctrl + C**.

## H. If something goes wrong

| What you see | What to do |
|---|---|
| `'python' is not recognized` | Use `py src/demo.py` instead. |
| `can't open file ... demo.py` | You are in the wrong folder. Do step C again. |
| Browser says **"This site can't be reached"** | The demo is not running yet, or was stopped. Check the terminal shows the `running offline at http://127.0.0.1:8765` line; if not, do step D again and wait. |
| The browser searches the web instead | You typed the address without `http://`. Type the whole thing: `http://127.0.0.1:8765` |
| `address already in use` / `Only one usage of each socket address` | A demo is already running in another terminal: use that one, or close it. Or start on another port with `python src/demo.py --port 8766` and open `http://127.0.0.1:8766` |
| The page shows **"Error: ..."** in red | Press Ctrl + C in the terminal, then do step D again. |
| The first click is slow | The models are still loading. Wait until the terminal shows the `running` line before clicking. |
| `The paging file is too small` or the laptop becomes very slow | Too many demos are open: each one loads a 1 GB model. Press Ctrl + C in **every** terminal running a demo (or close those terminals), then start **one** demo again. |

## I. Other offline checks (optional)

Run these in the same C4 folder (step C):

**Run the full pipeline on a real recording, with all network access blocked:**

    python src/pipeline.py --recording J26DS313_R0022 --offline-check --measure

→ look for `"network_attempts": 0` and `"leaks": 0`.

**Run all automated tests:**

    python -m pytest tests -q

→ look for `319 passed`.

## Before the panel

- Run **only one** demo at a time.
- Start it **2 minutes before you present**, so the models are loaded.
- Close Chrome tabs you don't need: the model needs about 1 GB of free memory.
- Keep this guide open in case anything goes wrong.
