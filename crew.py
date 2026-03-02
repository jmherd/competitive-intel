# crew.py
# Orchestrates the CrewAI multi-agent research pipeline.
# Manages task definitions, agent coordination, and structured output parsing.

import os
import json
import streamlit as st
from dotenv import load_dotenv
from crewai import Crew, Task, Process, LLM
from agents import (
    create_company_analyst,
    create_competitor_scout,
    create_news_analyst,
    create_synthesis_agent,
    CompanyProfile,
    CompetitorAnalysis,
    NewsAnalysis,
    IntelligenceReport
)

load_dotenv()


def build_company_task(agent, company_name):
    return Task(
        description=f"""Research {company_name} thoroughly using web search.
        
        Find and return accurate information about:
        - When the company was founded and where it is headquartered
        - Its core business model (how it makes money)
        - Its key products or services (list the top 3-5)
        - Its main strengths (what it does well)
        - Its main weaknesses (what it struggles with)
        - Estimated revenue or funding if available
        - Approximate employee count
        - Its market position (leader, challenger, niche player, etc.)
        
        Search for "{company_name} company overview", "{company_name} business model",
        and "{company_name} products revenue" to gather this information.
        
        Return your findings as a structured JSON object matching this exact format:
        {{
            "name": "company name",
            "founded": "year or date",
            "headquarters": "city, country",
            "business_model": "description",
            "key_products": ["product1", "product2", "product3"],
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2", "weakness3"],
            "estimated_revenue": "revenue figure or estimate",
            "employee_count": "number or range",
            "market_position": "description"
        }}""",
        agent=agent,
        expected_output="A JSON object containing the company profile"
    )


def build_competitor_task(agent, company_name):
    return Task(
        description=f"""Identify and research the top 3 competitors of {company_name}.
        
        First search for "{company_name} competitors" and "{company_name} vs alternatives"
        to identify who the real competitors are.
        
        Then research each competitor to find:
        - Their business model
        - Their key products or services
        - Their strengths vs {company_name}
        - Their weaknesses vs {company_name}
        - Their market position
        
        Also identify:
        - Where {company_name} has clear advantages over competitors
        - Where {company_name} is at a disadvantage
        - The overall competitive landscape dynamic
        
        Return your findings as a structured JSON object matching this exact format:
        {{
            "competitors": [
                {{
                    "name": "competitor name",
                    "business_model": "description",
                    "key_products": ["product1", "product2"],
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"],
                    "market_position": "description"
                }}
            ],
            "competitive_landscape": "overall description",
            "target_company_advantages": ["advantage1", "advantage2"],
            "target_company_disadvantages": ["disadvantage1", "disadvantage2"]
        }}""",
        agent=agent,
        expected_output="A JSON object containing competitor analysis"
    )


def build_news_task(agent, company_name):
    return Task(
        description=f"""Find and analyze recent news about {company_name}.
        
        Search for "{company_name} news 2025", "{company_name} latest developments",
        and "{company_name} announcement" to find recent coverage.
        
        For each significant news item found:
        - Summarize the headline and key facts
        - Assign sentiment: "positive", "neutral", or "negative"
        - Assign significance: "high", "medium", or "low"
        - Write a brief summary of why it matters
        
        Then assess:
        - The overall sentiment across all news (-10 to +10 scale)
        - Key themes appearing across multiple stories
        - The most important recent development in one paragraph
        
        Return your findings as a structured JSON object matching this exact format:
        {{
            "news_items": [
                {{
                    "headline": "headline text",
                    "sentiment": "positive/neutral/negative",
                    "significance": "high/medium/low",
                    "summary": "why this matters"
                }}
            ],
            "overall_sentiment": "positive/neutral/negative",
            "sentiment_score": 5,
            "key_themes": ["theme1", "theme2"],
            "recent_developments": "paragraph summary"
        }}""",
        agent=agent,
        expected_output="A JSON object containing news analysis"
    )


def build_synthesis_task(agent, company_name, context_tasks):
    return Task(
        description=f"""You have been provided with research on {company_name} 
        from three specialized analysts: a company analyst, a competitor scout, 
        and a news analyst.
        
        Using ONLY the information provided by those analysts, write a strategic 
        intelligence report that includes:
        
        - Executive Summary: 3-4 sentences capturing the most important insights
        - Market Opportunity: Where is the biggest opportunity for {company_name}?
        - Key Risks: What are the top 3-5 risks facing the company?
        - Strategic Recommendations: 3-5 specific, actionable recommendations
        - Competitive Advantage: What is {company_name}'s single most defensible advantage?
        - Outlook: Is the overall outlook "positive", "neutral", or "negative"?
        
        Be specific and direct. Every sentence must add value.
        
        Return your findings as a structured JSON object matching this exact format:
        {{
            "executive_summary": "3-4 sentence summary",
            "market_opportunity": "description of biggest opportunity",
            "key_risks": ["risk1", "risk2", "risk3"],
            "strategic_recommendations": ["rec1", "rec2", "rec3"],
            "competitive_advantage": "single most defensible advantage",
            "outlook": "positive/neutral/negative"
        }}""",
        agent=agent,
        expected_output="A JSON object containing the intelligence report",
        context=context_tasks
    )


def parse_agent_output(output_str, model_class):
    """
    Extract and parse JSON from agent output.
    Handles markdown code blocks and extra text around JSON.
    """
    text = output_str.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
        
        # Fix any fields that should be lists but came back as strings
        for key, value in data.items():
            if isinstance(value, str):
                # Check if the model expects a list for this field
                field = model_class.model_fields.get(key)
                if field and hasattr(field.annotation, '__origin__'):
                    import typing
                    if field.annotation.__origin__ is list:
                        # Convert string to single-item list
                        data[key] = [value]
        
        return model_class(**data)
    except Exception as e:
        print(f"Parse error for {model_class.__name__}: {e}")
        return None


def run_intelligence_crew(company_name, status_callback=None):
    """
    Run the full competitive intelligence pipeline.
    Returns structured results from all four agents.
    """
    # Set API key in environment for CrewAI
    api_key = ""
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    os.environ["ANTHROPIC_API_KEY"] = api_key

    # Configure LLM
    llm = LLM(
        model="claude-haiku-4-5-20251001",
        api_key=api_key
    )

    # Create agents
    company_analyst = create_company_analyst(llm)
    competitor_scout = create_competitor_scout(llm)
    news_analyst = create_news_analyst(llm)
    synthesis_agent = create_synthesis_agent(llm)

    # Create tasks
    company_task = build_company_task(company_analyst, company_name)
    competitor_task = build_competitor_task(competitor_scout, company_name)
    news_task = build_news_task(news_analyst, company_name)
    synthesis_task = build_synthesis_task(
        synthesis_agent,
        company_name,
        context_tasks=[company_task, competitor_task, news_task]
    )

    # Assemble crew
    crew = Crew(
        agents=[company_analyst, competitor_scout, news_analyst, synthesis_agent],
        tasks=[company_task, competitor_task, news_task, synthesis_task],
        process=Process.sequential,
        verbose=True,
        llm=llm
    )

    if status_callback:
        status_callback("🚀 Crew assembled — starting research...")

    result = crew.kickoff()

    results = {
        "company": parse_agent_output(
            str(company_task.output), CompanyProfile),
        "competitors": parse_agent_output(
            str(competitor_task.output), CompetitorAnalysis),
        "news": parse_agent_output(
            str(news_task.output), NewsAnalysis),
        "report": parse_agent_output(
            str(synthesis_task.output), IntelligenceReport),
        "company_name": company_name
    }

    return results