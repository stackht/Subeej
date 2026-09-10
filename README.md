# 🌾 Subeej AI — Agriculture Intelligence & Decision Support Platform

## 📌 Core Idea

The **main goal of this project is to help farmers make practical, data-grounded decisions around crop selection, seed choice, soil suitability, season fit, and yield guidance** based on their actual field conditions.

Farming guidance is often generic and disconnected from the reality of a farmer's land — soil type, moisture, irrigation, season, and district all matter. Subeej AI is a structured, agriculture-focused assistant that shifts the experience from a vague chatbot to a grounded decision-support system: it interprets user intent, extracts field details from the conversation, anchors answers in real dataset evidence, and returns recommendations with reasons.

## 🎯 Problem Statement

Indian farmers and field advisors face four connected challenges:

- **One-size-fits-all advice** that ignores local soil, season, and district conditions
- **Vague chatbot answers** that repeat the user's words back without practical guidance
- **Wrong crop/seed selection** for a field's soil, moisture, and irrigation profile, hurting yield and income
- **Information overload** — evaluating land quality, soil, season, and seed options together is hard to do alone

A static FAQ or a generic LLM reply is too slow and too shallow for these field decisions.

## 💡 Proposed Solution

The system combines conversation understanding, dataset grounding, rule-based agronomy, and LLM reasoning into one decision loop:

1. **Intent-aware chat**
   - Detects what the user needs: crop recommendation, seed selection, soil suitability, disease help, fertilizer guidance, yield estimate, or season fit
2. **Field detail extraction**
   - Pulls crop, soil type, season, irrigation, moisture, land quality, previous crop, location, and area from plain language — stored in a session memory across messages
3. **Dataset grounding**
   - A 10,000-row agriculture seed & land dataset anchors answers in real soil → seed → yield evidence matched by crop, district, and state
4. **Agronomy rules + LLM reasoning**
   - Crop-rule knowledge (season, soil, temperature) is combined with the LLM; if the LLM is unavailable, the rule engine still answers
5. **Structured, explainable output**
   - Every answer carries a recommendation, suitable conditions, why it fits, missing details, and a targeted follow-up instead of filler text
6. **Field intelligence tools**
   - Image upload and analysis, field maps, charts, and a 3D seed viewer

## ⚙️ System Architecture

### 🔹 Input Layer

- Farmer chat prompts
- Field image uploads
- Location / district inputs
- Dataset CSV (10,000 rows of seed, soil, season, yield records)

### 🔹 Processing Layer

- Intent detection and entity extraction (regex + rule based)
- Session memory and multi-turn context merging
- Dataset matching, filtering, and ranking (crop, seed, soil candidates)

### 🔹 Decision Layer

- Agronomy rule engine (`lib/agronomy`)
- LLM reasoning via local Ollama or Mistral API, with automatic fallback
- Fallback response builder with targeted follow-up questions

### 🔹 Output Layer

- Structured JSON chat responses
- Recommendation cards, condition lists, and reasons
- Charts (Recharts), field maps (Leaflet), 3D seed views (Three.js)
- Health and recommendation APIs for integrations

## 🧠 Technologies

- **Frontend:** Next.js, React, TypeScript, Tailwind CSS
- **Visualization:** Recharts, Leaflet / react-leaflet, Three.js / react-three-fiber
- **AI / Backend:** Ollama (local model serving), Mistral API (cloud fallback), custom recommendation engine in `lib/`
- **Data:** Python utilities for training, RAG, and data-processing workflows (`seed-intelligence-ai/`)

## 📊 Key Capabilities

- ✅ Intent-aware agronomy chat assistant
- ✅ Dataset-grounded crop, seed, and soil recommendations
- ✅ Session memory across turns for multi-step guidance
- ✅ Local-first LLM (Ollama) with cloud fallback (Mistral API)
- ✅ Works offline through the rule-based fallback engine
- ✅ Field maps, yield charts, 3D seed viewer, and image analysis

## 🚀 Expected Impact

- Practical crop and seed choices matched to soil, season, and district
- Better yield through correct seed–crop–soil fit
- Clear fertilizer and disease guidance at the right crop stage
- Traceable, explainable answers instead of black-box chatbot replies
- A foundation for data-backed field decisions across Indian farmlands

## 🔬 Future Scope

- Bundle a fine-tuned agronomy model (GGUF) for one-click local deployment
- RAG retrieval over larger crop and market datasets
- Multilingual and regional-language support
- Farmer feedback loop to improve recommendation confidence
- Real-time weather and market price integration

## 🏁 Conclusion

This project tackles Indian farming's most practical decision problem: **choosing the right crop, seed, and soil match for a given field**. By combining intent-aware conversation, dataset grounding, agronomy rules, and LLM reasoning, Subeej AI helps turn uncertain field details into actionable, explainable farming guidance instead of vague chat.

## 🛠️ Implementation Status (Working)

The repository contains a runnable implementation:

- `pages/` — Next.js routes and API endpoints (`/chat`, `/dashboard`, `/field-intelligence`, `/seed-intelligence`)
- `lib/` — chat controller, dataset grounding, agronomy rules, LLM client (Ollama + Mistral)
- `components/` — chat panel, maps, charts, cards, 3D seed viewer, image uploader
- `data/source/` — agriculture seed & land dataset (10,000 rows)
- `scripts/` — dev, build, export, and Python training helpers
- `outputs/` — Ollama Modelfile for the reasoner model

### Quick Start

```bash
npm install
cp .env.local.example .env.local
npm run dev
```

For LLM-powered chat, run Ollama (or set `MISTRAL_API_KEY`):

```bash
ollama pull qwen2.5:1.5b
# then set OLLAMA_MODEL=qwen2.5:1.5b in .env.local
```

Open `http://localhost:3000/chat`.

## 📍 Next Milestones

1. Provide a bundled, fine-tuned Ollama GGUF model for local deployment.
2. Deploy to a container platform (Railway/Fly/Render) with Ollama in the same box.
3. Expand the dataset with regional crops and market-backed seed varieties.
4. Add multilingual and regional-language support.

## 📄 License & Copyright

**Copyright (c) 2026 Nihal Mishra.**

Released under the **MIT License**. See the [LICENSE](LICENSE) file for details.

Large generated artifacts (GGUF model binaries, `.env.local`) are excluded from the repository to keep it source-first.