# Study Agent — three-minute demo

**Status: All four parts are complete.** This script demonstrates the completed
Part 4 Streamlit app and the backend built in Parts 1–3.

## Before the presentation

- Activate the virtual environment and run `streamlit run app.py`.
- Use a short PDF with selectable text and a topic you can verify. Biology notes
  containing ATP work with the sample prompts below; otherwise adapt the topic.
- Configure `GEMINI_API_KEY` in `.env` or enter it in the password field before
  screen sharing. Check internet, quota and access to the configured model.
- Rehearse each generation once. Gemini latency varies; the timing below is a
  presentation target. Prepare a second browser session with completed replies
  so you can switch to clearly labelled earlier results if requests run slowly.
- Optional: install `ddgs` beforehand only if you want to demonstrate web search.

## 0:00–0:25 — problem and upload

**Click:** Choose your PDF in the sidebar, then **Process notes**.
Point to the ready message with the number of text pages and chunks.

**Say:** “Students often have long PDF notes and little time to revise. Study Agent
turns those notes into questions, summaries and practice material. The PDF is
extracted into text, split into overlapping chunks, and indexed with TF-IDF.”

## 0:25–0:55 — grounded question and sources

**Type:** `What is ATP? Cite the relevant pages.`

**Click:** Expand **Sources (pages)** under the reply. Then expand **Agent steps**.

**Say:** “The agent searches my notes and uses Gemini to answer. These excerpts
show the PDF pages I can check. The badge shows the selected task and tool.
The steps show actions and results, not the model's private reasoning.”

## 0:55–1:25 — an interactive quiz

**Click:** **Quiz**. Pick answers to all five questions; deliberately choose one
wrong answer if you can identify it. Click **Check answers**.

**Say:** “The model returns structured questions with four options and an answer
key. The interface checks my choices locally and shows a score and explanations.
Selecting an option does not call Gemini again.”

Point to **Generate Quiz — ✓ Used** in the sidebar's **Agent tools** panel.

## 1:25–1:50 — revision modes

**Click:** **Flashcards**, then expand the first card to reveal its answer.
Click **Important topics** and tick one item.

**Say:** “Flashcards support recall before revealing the answer. The topic
checklist helps me track what I have revised during this session.”

## 1:50–2:10 — take-away summary

**Click:** **Summary**, then **Download summary (.txt)**.

**Say:** “I can take a short revision summary away as a text file. For long PDFs,
the generator samples excerpts across the document, so it can omit details.”

## 2:10–2:40 — demonstrate agent tool selection

**Type:** `Find ATP in my notes, then calculate 12 * 8.`

**Click:** Expand **Agent steps** and point to the actual tool sequence.

**Say:** “Gemini can choose several tools: search the notes, perform arithmetic
with a safe calculator, and then answer. The arithmetic result is 96. Tool
selection can vary, so the trace tells us what actually happened.”

If the trace contains `fallback_router`, say: “The Gemini tool loop was unavailable
for this request, so the recorded keyword fallback selected one task.” Do not
present fallback as a successful multi-tool run. Use `Calculate 12 * 8` if needed.

## 2:40–3:00 — architecture and close

**Say:** “Streamlit handles the interface, pypdf reads the PDF, scikit-learn
retrieves matching passages, and Gemini selects tools and generates study
material. The UI calls one backend entry point: StudyAgent.run. Current limits
are keyword search, one PDF and no OCR. Next steps are semantic embeddings,
scanned-PDF support and multiple documents.”

Point to the ASCII architecture diagram in the README if time permits.

## Recovery lines

- **No text:** “This file is scanned. OCR is a future improvement; I'll use a
  text-based PDF for this demo.”
- **API/key/quota/network error:** “The app keeps the error visible without
  crashing. I'll switch to results generated during rehearsal.” Label them as
  earlier results, not a live successful request.
- **Slow generation:** Move to the prepared session and keep the explanation brief.
- **No web support:** “Web search is optional. This demo focuses on the uploaded notes.”
