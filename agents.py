# agents.py
# Defines Pydantic output models and CrewAI agent definitions.
# Each agent has a distinct role, goal, and personality.

import os
import time
from pydantic import BaseModel
from typing import List
from crewai import Agent
from crewai.tools import tool
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# PYDANTIC OUTPUT MODELS
# ============================================================

class CompanyProfile(BaseModel):
    name: str
    founded: str
    headquarters: str
    business_model: str
    key_products: List[str]
    strengths: List[str]
    weaknesses: List[str]
    estimated_revenue: str
    employee_count: str
    market_position: str


class CompetitorInfo(BaseModel):
    name: str
    business_model: str
    key_products: List[str]
    strengths: List[str]
    weaknesses: List[str]
    market_position: str


class CompetitorAnalysis(BaseModel):
    competitors: List[CompetitorInfo]
    competitive_landscape: str
    target_company_advantages: List[str]
    target_company_disadvantages: List[str]


class NewsItem(BaseModel):
    headline: str
    sentiment: str
    significance: str
    summary: str


class NewsAnalysis(BaseModel):
    news_items: List[NewsItem]
    overall_sentiment: str
    sentiment_score: int
    key_themes: List[str]
    recent_developments: str


class IntelligenceReport(BaseModel):
    executive_summary: str
    market_opportunity: str
    key_risks: List[str]
    strategic_recommendations: List[str]
    competitive_advantage: str
    outlook: str


# ============================================================
# SEARCH TOOL
# ============================================================

@tool("Web Search")
def web_search(query: str) -> str:
    """Search the web for information about a company or topic."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            time.sleep(3)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))

            if results:
                formatted = ""
                for r in results:
                    formatted += f"Title: {r.get('title', '')}\n"
                    formatted += f"URL: {r.get('href', '')}\n"
                    formatted += f"Summary: {r.get('body', '')}\n"
                    formatted += "-" * 40 + "\n"
                return formatted
            
            # No results — wait longer before retry
            time.sleep(5)

        except Exception as e:
            time.sleep(5)
            if attempt == max_retries - 1:
                return f"Search error after {max_retries} attempts: {str(e)}"
    
    return "No results found after multiple attempts."


# ============================================================
# AGENT DEFINITIONS
# Each function accepts llm as a parameter passed in from crew.py
# ============================================================

def create_company_analyst(llm):
    return Agent(
        role="Company Research Analyst",
        goal="Research and profile the target company thoroughly and accurately",
        backstory="""You are a senior business analyst with 15 years of experience 
        profiling companies for investment firms. You are known for finding accurate, 
        specific information rather than making generalizations. You always search 
        for concrete data points: founding dates, revenue figures, employee counts, 
        and specific products. You never fabricate information — if you can't find 
        something, you say so clearly.""",
        tools=[web_search],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_competitor_scout(llm):
    return Agent(
        role="Competitive Intelligence Specialist",
        goal="Identify and analyze the top competitors of the target company",
        backstory="""You are a competitive intelligence specialist who has helped 
        Fortune 500 companies understand their competitive landscape for over a decade. 
        You excel at identifying who the real competitors are (not just the obvious ones), 
        finding their strengths and weaknesses, and understanding how they position 
        themselves in the market. You think in terms of competitive dynamics and 
        market positioning.""",
        tools=[web_search],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_news_analyst(llm):
    return Agent(
        role="Market News & Sentiment Analyst",
        goal="Find and analyze recent news and market signals about the target company",
        backstory="""You are a market analyst who specializes in news sentiment and 
        its impact on business performance. You've spent years reading between the lines 
        of press releases, identifying what news signals are genuinely significant versus 
        noise, and understanding how public perception affects business outcomes. You are 
        skilled at assigning accurate sentiment scores and identifying emerging themes 
        in news coverage.""",
        tools=[web_search],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_synthesis_agent(llm):
    return Agent(
        role="Strategic Intelligence Synthesizer",
        goal="Synthesize all research into actionable strategic intelligence",
        backstory="""You are a chief strategy officer with experience advising boards 
        of directors at major corporations. You take raw research and transform it into 
        clear, actionable strategic intelligence. You think in terms of opportunities, 
        risks, and recommended actions. You are direct, specific, and never pad your 
        analysis with filler. Every sentence in your reports earns its place.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )