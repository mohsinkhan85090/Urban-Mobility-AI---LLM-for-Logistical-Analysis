<p align="center">RESEARCH PROJECT</p>

<h1 align="center">Urban Mobility AI</h1>

<h2 align="center">LLM for Logistical Analysis</h2>

<p align="center">
A hybrid intelligence assistant combining <b>Large Language Models (LLMs)</b>, 
<b>Retrieval-Augmented Generation (RAG)</b>, and <b>structured analytical tools</b> 
to study urban transportation patterns using <b>NYC taxi trip data</b>.
</p>

<p align="center">


---

## 01. Research Motivation  

Urban transportation systems generate enormous volumes of operational data, yet turning that data into actionable insight typically requires both statistical reasoning and natural language interaction.  

This project investigates how language models can serve as an **interface layer over urban mobility datasets**.

Rather than relying on free-form LLM generation, the system uses a **hybrid design**:
- Analytical questions → answered from retrieved data  
- Computational questions → routed through structured tool execution  

This approach:
- Reduces hallucination risk  
- Improves interpretability  

###  Areas of Investigation:
- Urban logistics analysis  
- Demand-aware mobility reasoning  
- Natural language querying over transport datasets  
- Hybrid decision systems (symbolic tools + RAG)  
- Intelligent trip planning with real-time external context  

---

## 02. Key Features  

| | |
|----------|----------|
| ✦ Hybrid query handling with LLM-based intent routing | ✦ Retrieval-augmented generation over NYC taxi trip records |
| ✦ Tool-based computation for distance, fare, route, and trip planning | ✦ Zone and borough resolution for taxi location identifiers |
| ✦ Historical route statistics derived from trip-level data | ✦ Optional real-time traffic via Google Distance Matrix API |
| ✦ Weather-aware trip adjustment via OpenWeatherMap API | ✦ Modular Python architecture for experimentation and extension |

---

## 03 System Architecture  

```text
User Query Input
      │
      ▼
   LLM Router
      │
      ├── ANALYTICAL ───────► Vector Retrieval + RAG
      │                         │
      ├── COMPUTATIONAL ────► Structured Tool Execution
      │                         │
      └── HYBRID ───────────► Tool First → RAG Supplement
                                │
                                ▼
                    Natural Language Response
---

##  Query Routing  

| Type           | Execution Path                          | Example Query |
|----------------|----------------------------------------|--------------|
| ANALYTICAL     | Vector retrieval + RAG                 | "What zones have the highest demand on weekends?" |
| COMPUTATIONAL  | Direct tool invocation                | "How much is a taxi from Chelsea to LaGuardia?" |
| HYBRID         | Tool → RAG explanation                | "Plan my trip from SoHo to Times Square." |

---

##  Pipeline Steps  

1. **User Query Input**  
   Natural language queries such as:  
   _"How far is Midtown from JFK?"_  

2. **LLM-Based Router**  
   Classifies queries into:
   - ANALYTICAL  
   - COMPUTATIONAL  
   - HYBRID  

3. **Execution Path**  
   Routes query to the appropriate pipeline  

4. **Response Generation**  
   - Tool outputs → summarized  
   - RAG answers → constrained to retrieved evidence  
   - Final output → natural language response  

---

##  04. Project Modules  

### Core Files  

| File | Description |
|------|------------|
| `main.py` | Entry point for execution |
| `ask.py` | Orchestration: routing, tools, and RAG |
| `llm_router.py` | Intent classification and routing |
| `llm_layer.py` | LLM prompting and summarization |
| `retriever.py` | Retrieves documents from vector DB |
| `vector_store.py` | Builds Chroma vector database |
| `tool_registry.py` | Registers and validates tools |

---

### Tooling Layer  

| File | Description |
|------|------------|
| `tools/distance_tool.py` | Distance estimation using historical data |
| `tools/fare_tool.py` | Fare estimation using median pricing |
| `tools/route_optimizer.py` | Computes route duration and fare |
| `tools/urban_trip_planner.py` | Trip planning with traffic & weather |
| `tools/zone_resolver.py` | Resolves fuzzy zone names |

---

### External Services  

| File | Description |
|------|------------|
| `external_services/traffic_service.py` | Google Distance Matrix API |
| `external_services/weather_service.py` | OpenWeatherMap API |

---

## 05. Dataset  

The system uses **NYC taxi trip data** as its primary knowledge base.

### Usage:
- Powers the **RAG retrieval pipeline**  
- Provides **statistical estimates for tools**  

---

<p align="center"><b>Urban Mobility AI · Research Project · NYC Taxi Trip Data</b></p>
